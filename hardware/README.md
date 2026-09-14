# Feetech STS3215 calibration and teleoperation

These utilities use LeRobot's Feetech support. Bus diagnostics, leader
calibration, and MuJoCo teleoperation never command servo positions. Physical
follower teleoperation does command positions and must be tested progressively.

## Control and calibrate the FMC4030 CNC

The guarded CNC utility communicates with the controller at
`192.168.0.30:8088` by default. Test TCP connectivity without transmitting a
controller command:

```bash
.venv/bin/python hardware/control_fmc4030.py check
```

After installing the project in editable mode, the equivalent command is
`.venv/bin/hepha-cnc check`.

Read the current X/Y/Z positions, speeds, limit inputs, axis states, and homing
state without causing motion:

```bash
.venv/bin/python hardware/control_fmc4030.py status
```

Before any automatic travel, manually activate every negative and positive
limit switch one at a time and run `status` after each activation. Confirm that
the correct axis and direction changes to `TRIGGERED`. Do not commission the
machine if this mapping is wrong.

Home all axes sequentially toward their negative switches:

```bash
.venv/bin/python hardware/control_fmc4030.py home \
  --axes x y z \
  --direction negative \
  --speed 10 \
  --acceleration 20 \
  --backoff 5
```

### Actively find and record min, mid, and max

The endpoint calibration deliberately matches the simple manual procedure that
works with this controller. For one selected axis it:

1. sends one bounded relative move toward MIN;
2. waits for the calculated move duration plus a settling margin;
3. opens a fresh connection, reads status once (with read-only retries), and
   verifies that the axis is stopped and its negative limit is active;
4. moves 5 mm away from the switch;
5. repeats the same process toward MAX and backs away again; and
6. saves both safe endpoints and their midpoint together.

There is no status polling while an axis is moving, and the old incremental and
parallel calibration commands have been removed. Calibrate one axis at a time:

```bash
.venv/bin/python hardware/control_fmc4030.py \
  --status-timeout 3 \
  calibrate-axis \
  --axis x \
  --status-read-timeout 120
```

Run the same command with `--axis y` and `--axis z` to calibrate those axes. The
defaults reproduce the tested motion: 600 mm maximum travel in each direction,
20 mm/s speed, 200 mm/s² acceleration/deceleration, a 5 mm backoff, and a 2
second settling margin. You can override them explicitly, for example:

```bash
.venv/bin/python hardware/control_fmc4030.py calibrate-axis \
  --axis y \
  --max-travel 600 \
  --speed 20 \
  --acceleration 200 \
  --deceleration 200 \
  --backoff 5 \
  --settle-seconds 2 \
  --status-read-timeout 120
```

Results are updated atomically in
`hardware/fmc4030_limit_calibration.json`. Previously calibrated axes are
preserved, and nothing from the current axis is saved unless both limit checks
succeed. If a command, wait, or status check is interrupted, the utility also
requests an immediate stop for the selected axis.

`--status-timeout` is one interruptible TCP receive window. During calibration,
`--status-read-timeout` is the total deadline for each endpoint status read.
The default permits as many 3-second receive windows as fit within 120 seconds;
it no longer fails after only five slow windows.

Calibrate X, Y, and Z sequentially with one confirmation using:

```bash
.venv/bin/python hardware/control_fmc4030.py \
  --status-timeout 10 \
  calibrate-all-axes \
  --status-read-timeout 300
```

The default order is X, Y, Z and can be changed with `--order`, for example
`--order z x y`. Each axis completes its MIN and MAX sequence before the next
axis starts. All six endpoints are written atomically only after all three axes
succeed, preventing a partial run from mixing calibration coordinate frames.

### Move to named positions A and B

`goto-position` reads `hardware/fmc4030_limit_calibration.json` and derives the
requested absolute position from its current safe MIN, MID, and MAX values:

- A = `(X mid, Y min, Z max)`
- B(drawer) = `(X mid - delta_1, Y min + 50 mm + delta_2, Z min)`

Move to A:

```bash
.venv/bin/python hardware/control_fmc4030.py goto-position --position A
```

Position B requires a drawer number:

```bash
.venv/bin/python hardware/control_fmc4030.py goto-position \
  --position B \
  --drawer 9
```

The original drawer `delta_1` is inverted before being applied to X. The
drawer-row `delta_2` remains active relative to the new Y base:

| Drawer | delta_1 | Applied X offset | delta_2 | Applied Y |
| -----: | ------: | ---------------: | ------: | --------- |
| 1 | -50 mm | +50 mm | +50 mm | Y min + 100 mm |
| 2 | 0 mm | 0 mm | +50 mm | Y min + 100 mm |
| 3 | +50 mm | -50 mm | +50 mm | Y min + 100 mm |
| 4 | -50 mm | +50 mm | 0 mm | Y min + 50 mm |
| 5 | 0 mm | 0 mm | 0 mm | Y min + 50 mm |
| 6 | +50 mm | -50 mm | 0 mm | Y min + 50 mm |
| 7 | -50 mm | +50 mm | -50 mm | Y min |
| 8 | 0 mm | 0 mm | -50 mm | Y min |
| 9 | +50 mm | -50 mm | -50 mm | Y min |

The utility rejects a missing drawer for B, a drawer supplied for A, an incomplete
calibration, a controller-address mismatch, or any computed target outside a
safe calibrated interval before transmitting motion. It sends the absolute
axis commands back-to-back so they execute in parallel, then returns as soon as
the controller acknowledges them. Controller motion may continue after the
process exits. Named position commands also start immediately without an Enter
confirmation.

The slow initial status read is skipped by default. Add `--read-initial-status`
to verify that the axes are initially stopped, skip axes already at target, and
calculate a shorter wait from their current positions. This optional read does
not affect positioning mode: every movement packet remains absolute.
Add `--read-final-status` to perform the slow final stopped-state and position
verification. In that mode the utility waits using the largest calibrated axis
span before requesting status. Initial and final reads can be enabled
independently; both are skipped by default.

Defaults are 200 mm/s speed, 200 mm/s² acceleration, and 200 mm/s² deceleration.
Override them, the command transmission order, or status deadline when needed:

```bash
.venv/bin/python hardware/control_fmc4030.py \
  --status-timeout 3 \
  goto-position \
  --position B \
  --drawer 3 \
  --speed 10 \
  --acceleration 100 \
  --deceleration 100 \
  --order z x y \
  --status-read-timeout 300 \
  --read-initial-status \
  --read-final-status
```

After homing, move to any saved combination with `goto`:

```bash
.venv/bin/python hardware/control_fmc4030.py goto \
  --x min \
  --y mid \
  --z max
```

Axes move sequentially. The default order is Z, X, Y; change it for the
collision-safe order of the real machine with, for example,
`--order z y x`. The `goto` command accepts only `min`, `mid`, or `max` and
rejects a configuration created for a different controller address.

The raw homing endpoint does not acknowledge the transmitted command. After
sending it once, the utility monitors controller status until homing completes.
`--motion-timeout` is the maximum monitoring time (120 seconds per axis by
default); increase it for a long, slow axis, for example
`--motion-timeout 300`. The general `--timeout` option controls ordinary TCP
connections and motion-command acknowledgements. Status replies have a separate
3-second response window and make up to five read-only attempts; override it
globally with `--status-timeout` if this controller needs a longer window.

One persistent TCP connection is reused throughout each command workflow.
Before the first status request on a new connection, the client reads the
94-byte parameter block with `01 12`; this controller firmware otherwise
intermittently ignores a direct `01 03` request. The warm-up is read-only.
Status polling parses the documented 60-byte machine-status block immediately;
the unused 600-byte filename table is drained with a short bound and a delayed
table causes a clean reconnect instead of invalidating that status sample.
Read-only status transactions make exactly five total attempts before failing,
including connection establishment. Immediate single-axis stop is also retried
up to five times because repeating it is safe; the client briefly settles and
consumes an optional stop echo before polling again. Motion packets are never
retransmitted after sending: if their acknowledgement is lost, repeating a
relative move could duplicate the commanded travel.

If a status response is merely delayed, those receive attempts continue on the
same outstanding `01 03` request and TCP session. They do not repeatedly close
and reopen connections, which can strand this embedded single-client server on
an abandoned session. On process exit the socket uses an abortive close so the
firmware releases its single client immediately for the next CLI invocation.

Before motion, status may use all five short attempts. Once an axis is moving,
one missing status sample aborts the workflow and requests immediate stop; the
search never continues blindly until `--search-timeout`.

For a first packet-level physical test, clear the machine workspace, provide a
physical power cutoff, and request only a small, slow relative movement:

```bash
.venv/bin/python hardware/control_fmc4030.py move \
  --axis x \
  --position 1 \
  --mode relative \
  --speed 5 \
  --acceleration 10 \
  --deceleration 10
```

Every motion workflow shows its plan and pauses for Enter immediately before
the first command. A missing acknowledgement after transmission is reported as
an unknown machine state and is never retried automatically. Active limit
calibration uses the documented immediate single-axis stop packet; a physical
power cutoff is still required because software and Ethernet cannot provide an
independent emergency stop. No controller software-limit parameters are
changed, and MuJoCo limits are not treated as physical machine limits.

## Install Feetech support

The repository already uses `.venv`. Install the optional Feetech SDK once:

```bash
.venv/bin/python -m pip install "lerobot[feetech]==0.6.1"
```

The STS3215 bus must have the correct external supply connected. USB powers the
bus adapter, not the servos. Use the voltage specified on your servo model.

## Detect one bus, discover its IDs, and monitor positions

Connect and power one arm's bus, then run:

```bash
.venv/bin/python hardware/read_feetech_positions.py
```

The script:

1. finds likely USB serial adapters on macOS;
2. probes the Feetech baud rates and discovers responding servo IDs;
3. continuously refreshes a table with each servo's robot element, raw
   position, and position in degrees.

Press `Ctrl+C` to stop. Example output:

```text
Feetech STS3215 bus monitor
Port: /dev/cu.usbmodem101
Baud: 1,000,000   Servos: 12   Elapsed:     84.6 s

 ID  Robot element          Raw   Position
---  ------------------  ------  ---------
  1  shoulder right       831     73.04°
  2  shoulder left       3299    289.95°
```

For scrolling CSV output suitable for saving to a file, use:

```bash
.venv/bin/python hardware/read_feetech_positions.py --format csv
```

If both arm adapters are connected, select one explicitly:

```bash
.venv/bin/python hardware/read_feetech_positions.py \
  --port /dev/cu.usbmodem101
```

To verify the expected IDs and read at 20 Hz:

```bash
.venv/bin/python hardware/read_feetech_positions.py \
  --port /dev/cu.usbmodem101 \
  --ids 1 2 3 4 5 6 \
  --rate 20
```

To scan without starting the monitor:

```bash
.venv/bin/python hardware/scan_feetech_ids.py
```

Every servo on a bus must have a unique ID. Duplicate IDs cannot be reliably
distinguished because their replies collide on the shared wire.

## Calibrate each physical axis

The calibration utility captures three positions for one servo at a time:
MuJoCo minimum, home (`q=0`), and MuJoCo maximum. It disables torque only for
the axis currently being calibrated and never commands a position.

Support the arm securely, then run:

```bash
.venv/bin/python hardware/calibrate_feetech_positions.py
```

Follow the prompts to move each joint manually. Nine encoder samples are taken
at every pose and the median is saved. Progress is written after every completed
axis to `hardware/feetech_calibration.json`.

If calibration is interrupted, continue without repeating completed axes:

```bash
.venv/bin/python hardware/calibrate_feetech_positions.py --resume
```

To calibrate one or more axes separately, pass their IDs. Existing calibration
for every other axis is loaded and preserved automatically:

```bash
.venv/bin/python hardware/calibrate_feetech_positions.py \
  --ids 4
```

When both `--port` and `--ids` are supplied, calibration pings only the selected
IDs at 1,000,000 baud and skips exhaustive discovery. Use `--baudrate` for a
different known rate. If the quick ping fails, the utility automatically falls
back to the full scan.

If a selected ID is already calibrated, only that ID is replaced. Use
`--overwrite` only when you intentionally want to discard the complete existing
calibration and start a new file.

The finger joints use `q=0` for both minimum and home, so leave the finger
closed for both captures; maximum is the fully open pose.

Encoder endpoints are unwrapped jointly when a calibrated range crosses the
12-bit encoder zero or extends slightly beyond half a turn from home. A scale
difference between the two sides is reported as an advisory and does not force
the axis to be recaptured.

## Calibrate a physical follower arm

The follower uses a separate calibration because its encoder zero, horn
installation, and direction can differ from the leader. Connect the follower
through its own USB motor-bus adapter. Duplicate servo IDs are allowed across
the two arms because they are on separate serial buses.

Follower calibration captures the same three reference poses as leader
calibration, but orders them as `MIN`, `MAX`, then `HOME`. Ending at `HOME`
allows the utility to verify and commission the servo without asking for HOME
twice. Press Enter to capture each pose.

```bash
.venv/bin/hepha-follower-calibrate \
  --port /dev/cu.FOLLOWER_PORT \
  --ids 2 4 6 8 10 12
```

The default output is `hardware/feetech_follower_calibration.json`. In addition
to saving the three points, this utility persistently:

- centers the follower's captured home near raw encoder position 2048;
- selects position-control mode;
- writes hardware minimum and maximum position limits.

It never sends a goal position during calibration, so all calibration movement
is manual. Because homing and limit settings are written to the servo, verify
the selected port belongs to the follower before pressing Enter to begin.

As with leader calibration, one axis can be calibrated or replaced without
removing the others:

```bash
.venv/bin/hepha-follower-calibrate --ids 4
```

Supply both `--port` and `--ids` to use the same fast direct-ping path instead
of scanning every baud rate and servo ID.

Use `--resume` after an interruption. Re-run the interrupted axis because its
homing offset may already have been centered before the interruption.

## Teleoperate a physical follower

Always begin with one joint and `--dry-run`. The dry run disables torque on the
selected leader and follower axes, compares their calibrated joint angles, and
prints live targets without writing follower goals:

```bash
.venv/bin/hepha-follow \
  --leader-port /dev/cu.LEADER_PORT \
  --follower-port /dev/cu.FOLLOWER_PORT \
  --joints shoulder_l \
  --dry-run
```

When the readings and directions are correct, remove `--dry-run`:

```bash
.venv/bin/hepha-follow \
  --leader-port /dev/cu.LEADER_PORT \
  --follower-port /dev/cu.FOLLOWER_PORT \
  --joints shoulder_l
```

The leader and follower do not need to start aligned. The program seeds every
follower goal from that servo's present position before enabling torque, then
automatically approaches the current leader pose at 10 degrees per second by
default. Hold the leader still during this phase. Normal teleoperation begins
when every selected joint is within 5 degrees. Use `--startup-velocity-deg`,
`--startup-tolerance-deg`, and `--startup-timeout-seconds` to adjust this
transition. During operation it clamps joint limits, limits target motion to 30
degrees per second by default, and disables follower torque on exit, timeout,
or any communication error.

After testing axes individually, operate the complete left arm with:

```bash
.venv/bin/hepha-follow \
  --leader-port /dev/cu.LEADER_PORT \
  --follower-port /dev/cu.FOLLOWER_PORT \
  --joints shoulder_l forearm_l arm_l wrist_l hand_l finger_l
```

Use a physical emergency power cutoff and keep the follower workspace clear.
The leader and follower serial ports must be different.

## Run one physical teleoperation episode

The episode command continuously mirrors the physical leader onto the physical
follower while coordinating this CNC sequence:

1. command position A automatically;
2. wait for SPACE, then command position B for a randomly selected drawer;
3. wait for SPACE, then command position A again; and
4. wait for SPACE once more to finish the episode.

The selected drawer is printed prominently at startup. CNC packet transmission
runs in a background worker, so Feetech reading and follower commands continue
at the requested control frequency while the CNC is being commanded and moving.
SPACE is accepted only after a conservative CNC movement delay; slow controller
status reads are not used. Ctrl+C or an error disables follower torque and sends
an immediate stop request to all CNC axes.

```bash
.venv/bin/python hardware/teleoperate_episode.py \
  --leader-port /dev/cu.usbmodem58FA1019951 \
  --follower-port /dev/cu.usbmodem58FA1020401
```

After an editable install, the equivalent entry point is:

```bash
.venv/bin/hepha-teleop-episode \
  --leader-port /dev/cu.usbmodem58FA1019951 \
  --follower-port /dev/cu.usbmodem58FA1020401
```

Use `--drawer 1` through `--drawer 9` to test a fixed drawer, or `--seed`
to reproduce a randomized selection. All joints calibrated on both arms are
teleoperated by default; `--joints` can select a subset. This test version does
not write a dataset. Each control cycle is already represented as an
`EpisodeFrame` and sent through the `EpisodeSink` interface in
`hardware/teleop_episode.py`; a later LeRobot sink can store those frames and
CNC phase transitions without changing the hardware-control loop.

## Test calibrated axes in MuJoCo

After calibrating the left shoulder (servo ID 2), run:

```bash
.venv/bin/python hardware/teleoperate_mujoco_joint.py --id 2
```

The utility disables torque only on the selected physical servo, reads its
position, maps it piecewise through the captured minimum/home/maximum points,
and drives only the corresponding MuJoCo arm actuator. At startup, the MuJoCo
CNC axes are placed at `cnc_x=0`, `cnc_y=0`, and the `head_z` minimum
(`-0.1 m`). It never sends a position command to the physical servo. Close the
MuJoCo viewer or press `Ctrl+C` to stop.

To teleoperate several calibrated axes simultaneously, list their servo IDs:

```bash
.venv/bin/python hardware/teleoperate_mujoco_joint.py --ids 2 4
```

The utility reads all selected servos together and disables torque only for
those axes. Support every selected physical axis before confirming.

Encoder smoothing is enabled by default. To see unsmoothed values:

```bash
.venv/bin/python hardware/teleoperate_mujoco_joint.py --id 2 --smoothing 1
```

## Record MuJoCo episodes from the physical leader

The interactive recorder uses the calibrated, torque-disabled physical servos
as a leader while storing MuJoCo camera observations, simulated joint state,
and the exact actions applied to the simulator in LeRobot format. It runs in a
single process and asks whether to save, discard, retry, or quit after every
attempt.

```bash
.venv/bin/python -m hepha_lerobot.recording.teleop \
  --ids 1 2 3 4 5 6 7 8 9 10 11 12 \
  --repo-id hepha/mujoco_feetech_teleop \
  --root datasets/hepha_mujoco_feetech_teleop \
  --episodes 20 \
  --episode-seconds 60 \
  --fps 30
```

After reinstalling the editable project, the equivalent console command is
`.venv/bin/hepha-teleop-record`. The recorder fixes `cnc_x=0`, `cnc_y=0`, and
`head_z` at its minimum. It omits automatic IK task-phase labels; drawer and
task conditioning remain present.
