# Hepha

Hepha is a robotics project where I try to reproduce a simplified pipeline for building a humanoid-like robot.

The goal is to play with these technologies and give you a sense of how humanoid robots work.

The project builds a robot made of a CNC base and an upper body with robotic arms. The goal is to make it perform a simple warehouse-like task: place or remove a foam cube from drawers.

This document explains the full pipeline: CAD design, 3D printing, robot assembly, MuJoCo and Isaac Sim / Isaac Lab simulation, synthetic data generation with inverse kinematics, policy training, and later fine tuning with real world data.

To go further, I also explore a more complete company integration system around the 
robot. The idea is to connect the robots, the ERP system, and the RAG to a central coordinator, so that a company could control many robots from one chat while keeping the ERP and RAG updated live based on the robot actions.

Finally, I will share my perspective on the field, based on my experience: the promising areas of research, and the challenges that still remain.

This document was polished using AI, but the goal is to keep the text simple and close to my own words.

It is not a finished product or a review paper, so please forgive the lack of details. It is a practical project report about what I built, what I learned, and what still needs to be done.

PS: I made a summary video of the document here: **TODO: link to video**.

PS: The project is called **Hepha**, like Hephaistos, the Greek god of tools, 
craftsmanship, and invention.

![Hepha robot overview](assets/images/robot-overview.svg)

## Table of Contents

1. [Brief Words About Me](#brief-words-about-me)
2. [Before Starting](#before-starting)
3. [Why Robotics Is Hard](#why-robotics-is-hard)
4. [Pipeline](#pipeline)
5. [Robot Description](#robot-description)
6. [Task Description](#task-description)
7. [CAD Construction](#cad-construction)
8. [Simulation Data Collection](#simulation-data-collection)
9. [MuJoCo Simulation](#mujoco-simulation)
10. [Isaac Simulation](#isaac-simulation)
11. [Train A Policy On Simulated Data](#train-a-policy-on-simulated-data)
12. [MuJoCo Benchmark Policy](#mujoco-benchmark-policy)
13. [Fine Tune A Policy](#fine-tune-a-policy)
14. [Going Further](#going-further)
15. [Conclusion](#conclusion)
16. [My Personal Pipeline Of Humanoid Robots](#my-personal-pipeline-of-humanoid-robots)

## Brief Words About Me

I am a passionate machine learning engineer from Switzerland.

For the past five years I have played with robotics: Arduino, Raspberry Pi, CAD software, CNC systems including 3D printers, servo motors and stepper motors, actuators, radios, GSM modules, and, well, electronics in general.

I then used my machine learning background to train policies on the cloud and make hardware move intelligently.

And it is so much fun, you will see.

## Before Starting

If you don't have experience in robotics yet, I recommend you to have a 
look at the LeRobot project.

LeRobot is an open source project from Hugging Face that helps you build a 3D printed robot and run a machine learning policy on it.

**TODO:** add a GIF of my own LeRobot robot and a PushT policy demo.

![LeRobot demo placeholder](assets/gifs/lerobot-demo.svg)

I also encourage you to create your own policies in LeRobot. A custom policy that I 
really liked is the [DOT policy](https://github.com/IliaLarchenko/dot_policy) 
from Ilia Larchenko. It helped me well to understand the ACT policy and its 
limitations.

If you want to stay on your computer, without a 3D printed robot, you can also train policies in virtual tasks, for example the PushT task trained in a Gym environment.

**TODO:** add GIF of my PushT policy demo.

LeRobot has evolved from a small imitation learning library into one of the most 
complete open-source robotics frameworks. It now contains implementations of 
many SOTA models for imitation learning, reinforcement learning, 
Vision-Language-Action models, world models, and reward models.

I will mention and use some of these models: ACT, Diffusion models, VLA 
(Vision-Language-Action), and JEPA.

### Prerequisites For This Project

1. **CAD**: familiarity with CAD software such as Fusion360, FreeCAD or SolidWorks.
2. **Simulation engines**: MuJoCo from DeepMind, Isaac Sim from Nvidia.
3. **ML knowledge**: imitation learning (particularly behavior cloning) 
   reinforcement learning, ACT, diffusion models, VLA, JEPA.

## Why General Purpose Robotics Is Hard

Let me be clear: no, you cannot simply plug Claude Code, Gemini or ChatGPT into humanoid hardware and get a fully autonomous human-looking robot.

LLMs are very good at text, but the physical world is very different from text, and a lot more complex.

To get a sense of it, first note that for LLMs the amount of high quality training data available is incredibly large: the internet, billions of text examples. The space of English text to predict is relatively small: a few thousand common words or tokens.

General purpose robotics is very different.

The inputs are images, which are much more complex to analyze than text, possibly 
other sensor data: touch data, lidar data for depth, text, or audio commands from a human. Basically, any input your brain receives from your body.

The output is a set of servo joint coordinates, meaning actions of the robot in the physical world, and potentially voice if the robot should speak.

Unlike for LLMs, the amount of high quality robotics data, meaning sensor values and ground truth action pairs, is very limited.

Also unlike LLMs, where the prediction space is rather small and discrete, the action space in robotics is continuous, though it can be discretized, and immense.

As a result, even the best LLMs will not perform well if you simply plug them into your robot. It is also going to be very slow.

One needs models capable of understanding the mapping from world observations to 
actions, and vis-versa. I call them "world models".

Transformers (the main architecture of LLMs) might not even be the right 
architecture. To 
understand more why, see interviews and papers from Yann LeCun about world models:
- TODO: link interview NVIDA summit
- TODO: papers
He recently created AMI Labs, a start-up aiming to produce state of the art world 
models for real world applications.

To summarize, two important challenges of general purpose robotics today are:

1. **Data**: high quality ground truth observation-action pairs.
2. **Adequate models**: models capable of understanding the world and the surrounding physics, while still being very fast at inference.

### Data Challenge

To tackle the first challenge, good quality data, researchers use a mix of simulation and imitation learning episodes collected on the physical robot.

First, you train your robot in a virtual environment that mimics real physics, and hope the policy transfers to the physical robot. This is the sim-to-real problem.

Nvidia created Isaac Sim for exactly this purpose: to mimic real world physics accurately and create realistic simulations, running on Nvidia GPUs. MuJoCo serves a similar purpose. MuJoCo is more commonly used in research, less visually polished than Isaac Sim, but computationally more efficient because it can run on CPUs.

Simulation data is generally not enough to get a policy working outside the simulator. The policy usually needs to be fine tuned on episodes collected with the actual robot.

To collect these episodes, the robot must perform the task while a human or another system guides its movements. The robot being trained is called the follower. It records its observations and the corresponding ground truth actions.

This guiding process is called teleoperation. It can be done with another connected robot, called the leader, while the trained robot is the follower. It can also be done with a controller, although this becomes harder when the robot has many degrees of freedom.

**TODO:** add GIF of leader and follower + controller.

Teleoperation produces high quality ground truth data, but it is difficult to use it to collect millions of episodes. This is why simulation is used in the first place.

### Model Challenge

The research community is a bit divided on whether Transformer-like models can 
understand the world's physics well enough to be a solution for general purpose 
robotics.

Some big players are betting on scale: with enough data and compute, large models 
will be able to understand the 3D world.

Other researchers, like Yann LeCun, believe that current models are not designed 
to understand the world and the surrounding physics. They do not efficiently build useful representations of the world.

For example, to predict the trajectory of a ball from camera images, a Transformer may look at many irrelevant details in the image, such as the color of the sky or the background. This can be very inefficient.

World models are interesting because they seem closer to how the brain works. Before doing an action, for example fetching a spoon in the kitchen, your brain does not directly predict every muscle movement from the beginning to the end. It first plans at a high level: find the kitchen, find the spoon, come back. Then each of these steps can be decomposed into lower level actions. For "find the kitchen", this could mean stand up, rotate your body, scan the room, and walk toward the kitchen.

Each level requires a different type of intelligence. High level actions require reasoning and planning. Lower level actions are more mechanical, closer to reflexes, and must be very fast at inference. Also, before doing an important action, your brain can imagine your body doing it and predict what may happen without actually doing it. This is possible because the brain has some internal model of the world.

This is the intuition behind Yann LeCun's world model direction. He has proposed models like JEPA, Joint Embedding Predictive Architecture, as a possible foundation for world models. The idea is to learn useful representations of the world in an abstract embedding space, and to predict future states in this space instead of predicting every pixel directly. A robot could then reason inside this learned world representation before actually performing an action.

## Pipeline

Now that I went through the project overview and the main challenges of general 
purpose robotics, let's deep dive into the project and how I trained the policy:

```mermaid
flowchart TD
    A["1. CAD modeling<br/>Design the robot in Fusion360."]
    B["2. Simulation<br/>Build MuJoCo and Isaac Sim environments to validate the robot design and collect simulation data."]
    C["3. Leader and follower robots<br/>Build the 3D printed robots used later for real world data collection."]
    D["4. Policy training<br/>Train policies on simulation data."]
    E["5. Real world fine tuning<br/>Collect real world data with the follower and leader, then fine tune the policy on this data."]

    A --> B --> C --> D --> E
```

## Robot Description

The robot has 15 degrees of freedom.

The robot is composed of a CNC part and an upper humanoid body part with servo motors.

The CNC part is a CNC machine with stepper motors that I used to play with in 
previous projects:

![Robot overview placeholder](assets/images/robot-overview.svg)

The CNC part allows to place the upper humanoid body part in different work 
positions:

![CNC positions placeholder](assets/images/cnc-positions.svg)

## Task Description

The task resembles a realistic robot task in a warehouse-like environment. Here I call "warehouse-like environment" any environment composed of zones where robots maneuver and zones where products are stored, placed, or picked from. Examples are traditional warehouses, supermarkets, pharmacies, greenhouses, or even vineyards.

A robot would navigate in the environment and use its robotic arm to place objects into storage, or remove objects from storage.

For brevity, this report will not discuss autonomous navigation in the warehouse-like environment, as this does not necessarily require a machine learning model. If you are curious, the robot base for navigation looks like this:

![Mobile base placeholder](assets/images/mobile-base.svg)

In what follows, I will assume the task happens in a traditional warehouse composed of drawers, with objects to place into or remove from the drawers.

You can generalize this task to many use cases. In a greenhouse, it could mean placing seeds or harvesting a product from a rack. In a supermarket, it could mean placing products on a shelf, but not removing them, since this is done by customers.

For simplicity, I purposely chose not to focus on the upper part of the humanoid 
robot and not on the legs at the moment. This is an approach also used by Genesis 
AI (**TODO:** add video).

All in all, the task description is:

> Given a request from the user to place or remove a product in or from a 
> warehouse, the robot should move its body and arms to perform the task, where:
>
> 1. I use a foam cube to represent the product for simplicity.
> 2. I use a stack of drawers to mimic warehouse racks.

## Step 1: CAD Construction

Before building anything, I construct a 3D model of the robot.

This is required to validate the design, 3D print the robot, and build the simulation environment in MuJoCo and Isaac Sim.

For this I use Fusion360.

![CAD model placeholder](assets/images/cad-model.svg)

The path to the CAD project is **TODO**.

Each component is modeled as an assembly, possibly made of several sub-components:

```text
hepha-robot-cad
├── base
├── cnc_x
├── cnc_y
├── shoulder_l
├── shoulder_r
├── head
├── forearm_l
├── forearm_r
├── arm_l
├── arm_r
├── wrist_l
├── wrist_r
├── hand_l
├── hand_r
├── finger_l
├── finger_r
├── storage_rack
└── storage_bin
```

**TODO:** add GIF of the CAD.

![CAD GIF placeholder](assets/gifs/cad-preview.svg)

From this CAD, the robot components were 3D printed and assembled, both for the leader and follower.

Since I do not have a CNC machine for the leader, I will use a controller to lead 
the CNC follower as explained later.

**TODO:** add GIF of the real robot.

![Real robot GIF placeholder](assets/gifs/real-robot.svg)

## Step 2: Simulation

As mentioned above, the lack of data is one of the two main challenges in general 
purpose robotics. For this reason, and also to validate the hardware, I first built 
a simulation of the robot that I designed. I will try both MuJoCo from DeepMind and Isaac Sim from Nvidia to compare the two.

### MuJoCo

MuJoCo is widely used in research. It is very easy to install and has a low 
learning curve. It is known to be computationally efficient for physics 
simulation, while still simulating robot dynamics accurately. You can run it on 
your local machine, while Isaac Sim requires GPUs. One important drawback of 
MuJoCo compared with Isaac Sim is that it does not produce photorealistic 
rendering. This can be useful when the model requires camera frames as input 
(which is the case for humanoid robot models).

#### From CAD To Robot Description

To create the MuJoCo simulation, I will use the Fusion360 plugin called 
`ACDC4Robotics` to transform a CAD representation into `.urdf`, or even into an `.mjcf` file specific to MuJoCo. These are robot description files. They contain 
not only the visual meshes of the components, but also the joint information between robot components, inertia, friction, center of mass, etc. All of this is required to produce a simulation faithful to reality.

To use `ACDC4Robotics`, I had to transform my original `hepha-robot-cad` project into a new project called `hepha-robot-sim`.

In the CAD project, components are organized like robot parts. In the simulation project, they need to be organized like robot links: one component per link, and only one body inside each link component.

To do this, I first created an empty component for each link. Then I copied the relevant bodies into each link component, combined them into a single body, and finally added the correct joint between the links, either `slider` or `revolute`.

**TODO:** add path to the `hepha-robot-sim` file.

![MuJoCo visual placeholder](assets/images/mujoco-visual.svg)

#### Collision Geometries

The `.mjcf` file from the previous section only gives me visual mesh geometries with controllable joints. To obtain a physically realistic robot, I also need to activate physical collisions.

In MuJoCo, visual meshes are only used for visualization. The collision engine instead uses coarser and simpler geometries, which are faster and more numerically stable than arbitrary triangle meshes. These collision geometries are usually made from MuJoCo primitives such as boxes, spheres, and cylinders.

For each link in my updated Fusion360 `hepha-robot-sim` project, I had to create a 
coarse `.step` file made of primitive geometries. I decided to use only boxes. I 
then wrote a simple Python script to transform these `.step` files, made from simple 
Fusion360 extrusions, into an MJCF description of primitive box geometries.

I also made sure to have realistic center of mass and inertia in the final `.mjcf` 
robot description file.

**TODO:** add path of the collision `.step` files.

**TODO:** add GIF with collision geometries.

![Collision geometries GIF placeholder](assets/gifs/collision-geometries.svg)

#### Recording Episodes In Simulation

Now that I have a somewhat realistic digital twin of the robot, I can set things up to record virtual episodes of the robot doing the task. Three methods can be used to record episodes in the simulated environment:

1. use a controller,
2. use a physical leader,
3. use inverse kinematics, IK.

IK allows to compute the joint movements required to place the end effector, the robot hand with the gripper, in a target position. When used several times, it allows to artificially create a robot movement. For example, I can decompose the movement "grab the cube and place it into the drawer" into smaller targets: "place the hand in grab position", "close the hand", "move the hand above the drawer", and "open the hand".

With this technique, the overall movement of the robot is not very natural or 
flexible. For example, if the cube falls out of the hand, the robot will not 
re-fetch it and will continue the movement without the cube. But IK makes it 
possible to create a large set of episodes, which can be used to train a benchmark policy and fine tune it later with higher quality data.

Recording using a controller or a physical leader is on the other hand time 
consuming, so I decided to use IK first to train a benchmark policy.

I will use the physical leader and controller later to fine tune the policy. Starting from a strong benchmark policy is especially important for RL because it dramatically reduces exploration. It allows the agent to refine an already competent behavior instead of wasting time discovering basic skills from scratch.

**TODO:** add GIF of episodes.

![MuJoCo episode placeholder](assets/gifs/mujoco-episode.svg)

Before the start of each episode, I randomize the position and orientation of the 
cube, the colors of the geometries, and add a bit of noise to the camera position and orientation between episodes for better generalization.

Episodes are stored as a Hugging Face dataset using LeRobot's dataset format, also used by Nvidia and many robotics companies.

#### Training the policy




### Isaac Simulation

Same as MuJoCo.

**TODO:** complete this section later.

![Isaac Sim placeholder](assets/images/isaac-sim.svg)

### Train A Policy On Simulated Data

In this section I will use the simulated data from MuJoCo and Isaac to train a base Behavior Cloning policy.

Thanks to IK, I was able to produce thousands of episodes while trying to add some randomization to each episode.

Of course, keep in mind that this is not sufficient and is only meant to obtain a benchmark policy.

Imagine something unseen during training happens, for example the cube drops from the hand, or some drawers are randomly opened. Then the policy will likely fail because IK recorded episodes strongly lack natural randomness, which is sometimes hard to imagine beforehand.

Behavior cloning is a specific imitation learning method that learns a direct mapping from observations to actions using supervised learning.

Imitation learning is the broader field of learning behaviors from demonstrations, including behavior cloning and more advanced methods such as inverse reinforcement learning, DAGGER, and diffusion-based policies.

I will explore several Behavior Cloning methods, from standard models to foundation models.

I summarize each model in one sentence and invite the reader to ask its favorite AI model to learn more about these models:

- **ACT, Action Chunking Transformer:** predicts a sequence of future actions at once using a Transformer, producing smoother and more stable robot trajectories than single-action prediction.
- **Diffusion Policy:** generates robot actions through an iterative denoising process, allowing it to model multiple valid behaviors and produce robust, high-quality motions.

I also explore more complex models to open the work.

For example, VLA does not only take camera frames and robot joints as input, but also text describing the user task request, like "place the red cube in the upper left drawer":

- **Vision-Language-Action models:** learn a policy conditioned on visual observations, robot state, and natural language instructions, enabling a single model to perform many different tasks.
- **VLA-JEPA, World Models:** learn predictive latent representations of future world states, allowing the robot to reason about the consequences of its actions rather than directly imitating demonstrations.

By exploring models with fundamentally different learning paradigms, from direct action prediction to generative policies, language-conditioned foundation models, and predictive world models, I aim to develop a broad understanding of modern robot learning approaches and their respective strengths and limitations.

My dataset is made of 1000 generated episodes, split 90%-10% between train and test.

I trained on **TODO: GPU name, maybe RTX 5000?**

## MuJoCo Benchmark Policy

### ACT

**TODO:** add training command.

**TODO:** add Hugging Face model and dataset link.

**TODO:** add Weights & Biases link.

### Metrics

| Metric | Value |
| --- | --- |
| Success Rate (%) | TODO |
| Action Error (L1 / MSE / MAE) | TODO |
| Collision Rate | TODO |
| Inference Speed (Hz or ms/action) | TODO |
| Number of Demonstrations | TODO |

### Qualitative Results

**TODO:** add test ground truth GIF.

![Ground truth rollout placeholder](assets/gifs/test-ground-truth.svg)

**TODO:** add test predicted GIF.

![Predicted rollout placeholder](assets/gifs/test-predicted.svg)

## Fine Tune A Policy

### PPO, Proximal Policy Optimization

**TODO:** complete this section later.

Notes:

- Update repo to normal.
- Ask to add model to W&B and save model to Hugging Face.
- Ask for the command for training ACT.

## Going Further

Maybe for a next video: the project Hepha for companies with warehouse-type environments.

Combine with chatbot, query builder, and RAG.

## Conclusion

Recently some startups have paid people in India to record day to day laundry task episodes.

Still models and setup very specific to the task. But better than if then else 
from what used to be done! Not yet one model fits all like for LLMs.

And if you want to get inspired by the future of robotics and by how these technologies may impact daily life.

For this, go have a look at videos of Tesla Optimus, Figure AI, or European 
start-ups like Genesis AI, Mimic, UMEA, 1X or Neura Robotics.

It is very impressive and inspiring.

**TODO:** add video links.

Studying this field also makes me appreciate the complexity of the human brain and body, which is a masterpiece of engineering from nature.

Though no physical law prevents humans from copying it artificially, and this will happen sooner or later.

Industry is progressing fast, but humanoid robots are not in homes yet.

There has been huge progress, but there is still a long way to go.

Your phone video can now become a voxel world.

`depth-anything.cpp` is an open-source repo for local geometry inference.

It runs Depth Anything 3 without a Python stack.

Also talk about latest sensing sensors.

Some words on LLMs and why they are not necessarily the solution for robotics.

Likely there is not only one way to general purpose robotics: better simulations, better physics engines, more compute, better models, and better hardware.

## My Personal Pipeline Of Humanoid Robots

1. Simulation is required. Imagine a CAD specific to robotics, with an AI able to create any realistic 3D object in the CAD with all physical properties. Create a metaverse world where robots are trained. AI understanding 3D space is fundamental.
2. Models should be very efficient at learning, and inference should be very fast. Big big models are not necessarily good.
3. Very good hardware with many sensors.
4. Fine tune with real world data.
5. Fine tune with RL in the real world.

## References

This section is still a work in progress.

See [references/README.md](references/README.md).
