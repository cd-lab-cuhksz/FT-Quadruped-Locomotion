#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <iostream>
#include "GLFW_callbacks.h"
#include "MJ_interface.h"
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/jacobian.hpp"

// MuJoCo load and compile model
char error[1000] = "Could not load a1 model";
mjModel *mj_model = mj_loadXML("../models/unitree_a1/scene.xml", 0, error, 1000);
mjData *mj_data = mj_makeData(mj_model);

//************************
// main function
int main(int argc, const char **argv)
{
    pinocchio::Model a1_model;
    std::string urdf_pathIn = "../models/unitree_a1/urdf/a1.urdf";
    pinocchio::urdf::buildModel(urdf_pathIn, a1_model);
    std::cout << "model name: " << a1_model.name << std::endl;
    pinocchio::Data a1_data(a1_model);

    // ini classes
    UIctr uiController(mj_model, mj_data);        // UI control for Mujoco
    MJ_Interface mj_interface(mj_model, mj_data); // data interface for Mujoco
    DataBus RobotState(a1_model.nv);              // data bus

    /// ----------------- sim Loop ---------------
    double simEndTime = 20;
    mjtNum simstart = mj_data->time;
    double simTime = mj_data->time;

    // init UI: GLFW
    uiController.iniGLFW();
    uiController.enableTracking(); // enable viewpoint tracking of the body 1 of the robot
    uiController.createWindow("time_sim", false);

    mju_copy(mj_data->qpos, mj_model->key_qpos, mj_model->nq * 1); // set ini pos in Mujoco

    while (!glfwWindowShouldClose(uiController.window))
    {
        // advance interactive simulation for 1/60 sec
        //  Assuming MuJoCo can simulate faster than real-time, which it usually can,
        //  this loop will finish on time for the next frame to be rendered at 60 fps.
        //  Otherwise add a cpu timer and exit this loop when it is time to render.
        simstart = mj_data->time;
        while (mj_data->time - simstart < 1.0 / 60.0)
        {
            mj_step(mj_model, mj_data);

            simTime = mj_data->time;
            printf("-------------%.3f s------------\n", simTime);
            mj_interface.updateSensorValues();
            mj_interface.dataBusWrite(RobotState);

            std::cout << "Motor positions (DataBus): ";
            for (double pos : RobotState.motors_pos_cur)
                std::cout << pos << " ";
            std::cout << std::endl;
        }

        if (mj_data->time >= simEndTime)
        {
            break;
        }

        uiController.updateScene();
    }

    // free visualization storage
    uiController.Close();

    return 0;
}