Post

If you read this document it means that you also you are interested in robotics. I also made a video here (link to video). This document was polished using AI (using Codex in this repository for both the redaction and the code).

Abstract:
This document summarizes my research project and some of what I learned and turned to be important to build general purpose robots (like humanoid robots). I also give my personal opinion on the field based on my experience: the promising areas of research and the challenges. 
This document is not a review paper, so please forgive my lack of ..., I mean to give the intuition. 

Brief words about me: I am a passionate machine learning engineer from Switzerland. For the past 5 years I played with robotics: Arduino, Raspberry Pi, CAD softwares, CNC systems (including 3D printers), servo motors and stepper motors, actuators, radios and GSN modules, electronics in general. Iapplied my Machine Learning knowledge (training models and policies on cloud GPUs) to make these physical elements move intelligently in the physical world. And it's so much fun you will see! 

First, if you are reading this document without robotics knowledge, I strongly recommend you to first have a look at LeRobot project. It's an open source project from HuggingFace which allows you to very easily get a 3D printed robot setup and running a ML policy (add a GIF of your own Lerobot robot and Push T policy). I encoruage also to maybe play wjth your own policy like done by Lachenko. LeRobot has evolved from a small imitation learning library into one of the most comprehensive open-source robotics frameworks. It now contains implementations of imitation learning, reinforcement learning, Vision-Language-Action (VLA), world, and reward models. We will briefly mention some of these models (ACT, VAT, JAP) below. 

Prerequisite:
1. CAD: familiar with CAD softwares such as Fusion360, FreeCAD or SolidWorks
2. Simulation engines: Mujoco from Deep Mind, Isaac Sim from Nvidia
3. ML knowledge: immitation learning versus reinforcement learning, ACT, VAT, JAP

Before deep diving into my project and the different technologies I suggest you to get inspired about the future of robotics and what these technologies are and how will they impact our lives. For this, go have a look at videos of TESLA's Optimus, Figure AI, or European companies like Genesis AI, UMEA, 1X or Neural Robotics. It's very impressive!

The project aims to play with these technologies to give you a sense of how complex looking humanoid robots work.

Introduction:
Before continuing let me be clear: no you cannot simply plug in Claude code, Gemini or ChatGPT to a humanoid hardware to get a fully autonomous human looking robot. LLMs are very good at text, but the physical world is very different than text - and a lot more complex. To get a sense of it, note first that for LLMs the size of the high quality training data available is incredibly large (all internet, billions of text) and the size of the English language text to predict is relatively small (a few thousand words). 
On the other hand general purpose robotics is very different. Input are images (camera frames), possibly other sensor data (touching data, Lidar data for depth, etc), possibly text or audio (e.g. commands from a human) - basically anything your brain receives from your body. The output is a set of servo joint coordinates (actions of the robot in the physical world) and potentially voice (the robot speaking). Unlike for LLMs the amount of high quality data (sensor values and ground truth action pairs) is very limited. Unlike for LLMs for which the prediction space is rather small and discrete, the space of actions is continuous (though it can be discredized) and immense. As a result even the best LLMs will not perform well if you simply plug it to your robot (and it is gonna be very slow). One needs models capable of understanding the world to action mapping (aka world models). LLMs with Transformers might not even be the right approach (see interviews and papers from Yann LeCun about world models, who recently created his startup AMI aiming to produce state of the art world models for real world application). To summarizes the challenges, first we do not have as many data and second the models might require to be different. These are the two main challenges of robotics today. 

To tackle the first challenge (good quality data) researches use a mix of simulation and real life immitation learning episodes. 
First, you train your robot in a virtual environment which mimics the physics of the real world. You train your policy or model inside this environment and hope it transfers well to the real world (aka Sim to Real problem). NVidia created Isaac Sim exactly for that: aiming to mimic real world physics very accurately to create real world simulation (like a super real video game using Nvidia GPUs). 
Then, to fine tune the policy or model trained in simulation (or sometimes to train directly a policy from scratch by scipping the simulation part which some companies do, see summary sketch below) one use immitation learning on actual episodes from the real-world (with a leader and follower, or using a VR headset for example, see pictures below). These real world episodes record real world sensor values to action pairs for specific tasks (for example folding laundry). Recently some startup pay people in India to record day to day laundry tasks episodes (see video). 

The other challenge is the model. Some still believe in Transformer like models to brute force the problem (see ACT pokicy or VAT). Some researches like Yann Lecun believe that current models are not designed to understand the world and the physics ruling the world and do not create efficiently embedding of the world. For example to predict the trajectory of a ball given camera images the Transformer will look at all details on the image including the color of the sky or the background which is very inefficient. Some of Yann Lecun models for worl model are JAP.

Method:
In this project I tried to reproduce what a pipeline to build a robot humanoid looks like. The project is called Hepa (like Hephaistos, the god of tools in greek mythology). 

The pipeline looks as follows:
1. CAD modeling on Fusion360
2. 3D print and construct the leader and follower robot (we will also try to record episodes using a controller just to see how well it works. One could also a VR headset but this is too expensive for me for what it might bring). 
3. Collect simulation data (we will try to use both Mujoco and Isaac Sim to see the pros and cons of each). 

4. Train and fin tune policies on simulation data

5. Collect real world data using follower - leader

6. Fine tune the policy obtained from simulation data on real world data

7. Test and measure the performance of the final policy on the real robot

Robot description:
The robot is composed of a cnc part (A CNC machine with stepper motors that I used to play with in previous projects) and an upper humanoid body part with servo motors. See image below:

The CNC part allows to place the upper humanoid body part in different work positions, see image below:



Task description:
The task ressemble a realistic task in a warehouse like environment (e.g. a warehouse, a supermarket, a pharmacy, or even a greenhouse or a vineyard) where a robot would navigate in a an environment and use the robotic arm to place or take objects in or out of the warehouse like environment. For brievety consideration, this report will not discuss robot autonomous navigation in the warehouse environment as this does not necessarly require ML model (but don't take me wrong, in practice humanoid robots do use the policy for navigation as well). If you are curious, the robot base for navigating the robot looks like this:

Now, in what follows we will assume the task to happen in a warehouse composed of drawers with objects to place or to remove from the drawers (but again you can generalize this task to many use cases, e.g. for a greenhouse it would be placing (seeding a plant) removing (harvesting a product) from a rack. Another example: in a supermarket it would be placing products in a rack (but not removing them since this is done by customers going in the supermarket). For simplicity I purposely chose not to focus on the upper part of the humanoid robot and not on the legs at the moment, an approach also used by Genesis AI (see video). 

All in all, the task description is:
"Given a request to place or remove a product (we decide to use a foam cube for simplicity) in or from a warehouse (we use a stack of drawers to mimic wahreouse racks), the robot should move its body and arms to perform the user's requested task"

CAD construction:
Before building anything we construct a 3D model of the robot. This is required to validate the design, 3D print the robot and build the simulation environment (Mujoco and Isaac Sim). For this we use Fusion360:


The path to the CAD project is ... where each componenet is modeled as an assembly (possibly of several sub components).


GIF of the CAD

From this CAD, the robot components were 3D printed and assembled, both for the leader and follower (since I do not have a CNC machine for the leader, I will use a controller as explained in part x).

GIF of the real robot

Simulation Data Collection 

It is often dangerous (for the hardware) and time consuming to directly record real world episodes of the robot. As mentionned above, the lack of data is one of the 2 main current challenges in general purpose robotics. For this reason, and to validate the hardware, we will build a simulation of the robot that we designed. We will try both Mujoco (from Deep Mind) and Isaac Simulation (from Nvidia) to compare the two.

Mujoco Simulation: 

Mujoco is widely used in research. It is very easy to install and has a low learning curve.  It is known to be computationally efficient for phyisc simulation while simulating robot dynamics very accurately. You can run it on your local machine (while Isaac Sim requires GPUs). One important drawback of Mujoco versus Isaac Sim is that it does not produce photorealistic rendering which can be useful when the model requires camera frames (which is the case for humanoid robot models). 

To create the Mujoco simulation we will use the Fusion360 plugin called ACDC4Robotics to transform a CAD representation into URDF (or even MJCF file specific to Mujoco). These are files description for robots, not only containing visual meshes of the components but also the joints informations between robot components, inertia, friction, center of mass, etc. All of this required to produce a simulation faithfull to the reality. 
 
Using ACDC4Robotics requires to create a SIM file project created from the CAD project by creating one component per link (and not one component per robot part) and each link has only one body. So basically, I first created an empty component for each link, then copy pasted the bodies inside each robot link component, combined the bodies (if there are many) to have only one body per robot link component and finally adding the adequate joint ("slider" or "revolut") betweeb each robot link. 

One Mujoco I obtain:


However, by doing this only we only get a robot with visual mesh geometries for which joints can be controlled. To obtain a physically realistic robot, we also need to activate physical collisions. In Mujoco, the collision engine does not use the visual meshes for collision (visual meshes are only here for visualisation). Instead, the simulation engine use coarser / simpler geometries than the visuals to accelerate collision computation. These coarser geometries need to be made for Mujoco's primitive geometries, simple analytical shapes used for collision detection, physics. They are much faster and more numerically stable than arbitrary triangle meshes. Examples are: box, sphere and cylinder. So for each link in my updated Fusion360 SIM project we had to create an coarse STEP file made of primitive geomtries (we decied to use only boxes) - we designed a simple Python code to transform a STEP file made to simple Fusion360 extrusion into a MJCF description of primitive box geometries. 

See the path of the collision STEP files.

GIF with robot vizualisation only

GIF with collision geomtries

We also made sure to have realistic center of mass and intertia in our final MJCF robot file description.
Now that we have a somehwat realistic digital twin of our robot, let's set things up to record virtual episodes of the robot doing the task we setted up (aka task description above). 
Two methods can be used to record episodes in the simulated environment in order to train a first simple benchmark policy for the task:
- Use a controller
- Use a phyisical leader
- Use inverse kinematics (IK). IK allows you to compute the joint movements required to place the end effector (the robot hand with the gripper) in a target position. Hence when used several times it allows to artifically create a robot movement (e.g. one IK to place the gripper over the cube, one IK to open the drawer, one IK to place the cube of the open drawer, etc.). With this technique, the overall movement of the robot is not natural or flexible. For example, if the cube falls out from the hand the robot will not re fetch the cube and might even continue its movement without the cube. But it i allows to create a very large set of episodes to create a base / benchmark policy to be fine tuned later using higher quality data. 

Recording using a controller or a physical leader is time consuming so we decided to use IK first to train a becnhmark policy. We will use the physical leader and controller later to fine tune the policy. Starting from a strong benchmark policy is especially important for RL because it dramatically reduces exploration, allowing the agent to refine an already competent behavior instead of wasting time discovering basic skills from scratch.

Episodes obtained from IK looks like below. We randomize the position and oriention of the cube and the colors of the geometries for better generalization. For the same reason, we also add a bit of random noise to the position and orientation of the camera between epsiodes. 

GIF of episodes


Episodes are stored as a HuggingFace dataset using LeRobot's dataset format (also used by Nvidia and many robotic companies).


Isaac Simulation:

Same as Mujoco


Train a policy on simulated data:

In this section we will use the simulated data from Mujoco and Isaac to train a base Behavior Cloning (BC) policy. Thanks to IK we were able to produce thousends of episodes while trying to add some randomization to each episode. Of course, keep in mind that this is not sufficient and is just meant to obtain a benchmark policy. Imagine something unseen is in the training is observed (e.g, the cube drops from the hand, or some drawers are randomly opened) then the policy will likely fail because IK recorded episodes strongly lack of natural randomness (sometimes hard to image beforehand). 

Calude: "Behavior cloning is a specific imitation learning method that learns a direct mapping from observations to actions using supervised learning, whereas imitation learning is the broader field of learning behaviors from demonstrations, including behavior cloning and more advanced methods such as inverse reinforcement learning, DAGGER, and diffusion-based policies."

We will explore several Behavior Cloning (BC) methods from standard to foundation models. We summarize each model in one sentence and invite the reader to ask its favorite AI model to learn more about these models:

* ACT (Action Chunking Transformer): Predicts a sequence of future actions at once using a Transformer, producing smoother and more stable robot trajectories than single-action prediction.
* Diffusion Policy: Generates robot actions through an iterative denoising process, allowing it to model multiple valid behaviors and produce robust, high-quality motions.

To open the work:
* Vision-Language-Action (VLA) models: Learn a policy conditioned on visual observations, robot state, and natural language instructions, enabling a single model to perform many different tasks.
* VLA-JEPA (World Models): Learns predictive latent representations of future world states, allowing the robot to reason about the consequences of its actions rather than directly imitating demonstrations.

By exploring models with fundamentally different learning paradigms—from direct action prediction to generative policies, language-conditioned foundation models, and predictive world models—we aim to develop a broad understanding of modern robot learning approaches and their respective strengths and limitations.


Mujoco Benchmark Policy:


ACT: Overall Metrics: (Success Rate (%),  Action Error (L1/MSE/MAE), Collision Rate, Inference Speed (Hz or ms/action), Number of Demonstrations) Test Grount Truth GIF , Test Predicted GIF





Fine tune a policy:

PPO (Proximal Policy Optimization)




Short summary of current models:






GOING FURTHER (maybe for a next video): the project Hepha for companies with warehouse type environment. Combine with Chatboat, Query Builder, RAG



Concusion:

Studying this field also allows us to appreciate the complexity of our brain and body - a chief of engineering of mother nature. Though no physical low prevent human from copying it artificially, and this will happen sooner or later.
Industry good - not humanoid robot in home yet. Super advancement but still long way to go.

Your phone video can now become a voxel world.
depth-anything.cpp is an open-source repo for local geometry inference.
It runs Depth Anything 3 without a Python stack.
Also talk about latest sensing sensors
Some words on LLM and why there are not necessarly the solution for robotics.
Likely there is not only one way to general purpose robotics: better simulations (physics engine, compute, ), better models, better hardware. 

