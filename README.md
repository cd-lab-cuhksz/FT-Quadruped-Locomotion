# FT-Quadruped-Locomotion

## FT-QuadLo: An Optimization-based Fault-Tolerant Control Framework for Quadruped Locomotion

## Environment

**Suggesting**

- OS: Ubuntu 22.04.4 LTS x64
- Compiler：g++ 11.4.0, python 3.9.21

**Dependence**

This repository is based on MuJoCo for simulation testing of the Unitree A1 robot. It includes MuJoCo's simulation engine, the Crocoddyl solver, Pinocchio dynamics library, Eigen, Quill logging tool, GLFW graphics library, JsonCpp parsing library, and other components. However, the simulation interface requires OpenGL support and must be installed on the system.

```Bash
# Update & Install Dependencies
sudo apt-get update
sudo apt install git cmake gcc-11 g++-11
sudo apt install libglu1-mesa-dev freeglut3-dev
```

**Demo**


https://github.com/user-attachments/assets/66cfee15-6a36-4080-bc77-e11ab5094848

