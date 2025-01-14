# Isaacsim-UR5

## Introduction
This repository provides a custom Reinforcement Learning (RL) environment using the UR5 robotic arm and the Robotiq 2F-140 gripper. The environment is designed to accept visual information as input, allowing for the development and testing of vision-based RL algorithms. It integrates NVIDIA Isaac Sim for realistic simulation, providing a robust platform for robotics research and experimentation.

## Prerequisites
Before starting, make sure you have the following installed:

- NVIDIA Isaac Sim 4.2.0
- `cmake` and `build-essential` packages

You can install the required packages with the following command:

```bash
sudo apt install cmake build-essential
```

## Installation
First of all you should install **IsaacSim** at [Installation Guide of IsaacSim](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html).
I recommend you to use conda virtual environment for this repository. 

```bash
mkdir ~/workspace
cd workspace
git clone git@github.com:jmSNU/IsaacSim-UR5.git
cd IsaacSim-UR5/IsaacLab
sudo apt install cmake build-essential
./isaaclab.sh -i
export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac-sim-4.1.0"
```

## Recommended Protocol
```bash
cd $ISAACSIM_PATH
source setup_conda_env.sh && source setup_python_env.sh
conda activate "conda_virtual_env"
cd ~/workspace/IsaacSim-UR5
python IsaacLab/source/standalone/tutorials/00_sim/create_empty.py
```

## Environments
- [Isaac-UR5-Reach-v0](https://github.com/jmSNU/Isaacsim-UR5/tree/main/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5_manipulation/ur5_reach)
- [Isaac-UR5-Push-v0](https://github.com/jmSNU/Isaacsim-UR5/tree/main/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5_manipulation/ur5_push) 
- [Isaac-UR5-Lift-v0](https://github.com/jmSNU/Isaacsim-UR5/tree/main/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5_manipulation/ur5_lift)
- [Isaac-UR5-Pick-v0](https://github.com/jmSNU/Isaacsim-UR5/tree/main/IsaacLab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/ur5_manipulation/ur5_pick)
