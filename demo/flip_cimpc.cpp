#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>
#include "GLFW_callbacks.h"
#include <string>
#include <iostream>
#include "pinocchio/parsers/urdf.hpp"
#include "pinocchio/algorithm/jacobian.hpp"

#include "MJ_interface.h"
#include "pd_fft_ctrl.h"
#include "data_logger.h"
#include "data_bus.h"
#include "joystick_interpreter.h"
#include "cimpc.h"

const double dt = 0.001;
// const double dt_40Hz = 0.025;
const double dt_Hz = 0.025; // 40Hz
// const double dt_Hz = 0.1; // 40Hz
// MuJoCo load and compile model
char error[1000] = "Could not load a1 model";
mjModel *mj_model = mj_loadXML("../models/unitree_a1/scene.xml", 0, error, 1000);
mjData *mj_data = mj_makeData(mj_model);

//************************
// main function
int main(int argc, char **argv)
{
    std::string urdf_pathIn = "../models/unitree_a1/urdf/a1.urdf";
    std::string srdf_pathIn = "../models/unitree_a1/srdf/a1.srdf";

    // initialize classes
    UIctr uiController(mj_model, mj_data);        // UI control for Mujoco
    MJ_Interface mj_interface(mj_model, mj_data); // data interface for Mujoco
    CIMPC cimpc(0.025, urdf_pathIn, srdf_pathIn); // CIMPC solver
    // CIMPC cimpc(0.08, urdf_pathIn, srdf_pathIn);                                // CIMPC solver
    DataBus RobotState(cimpc.nv);                                               // data bus
    PVT_Ctr pvtCtr(mj_model->opt.timestep, "../config/joint_ctrl_config.json"); // PVT joint control
    JoyStickInterpreter jsInterp(mj_model->opt.timestep);                       // desired baselink velocity generator
    DataLogger logger("../record/datalog.log");                                 // data logger

    // initialize UI: GLFW
    uiController.iniGLFW();
    uiController.enableTracking();                  // enable viewpoint tracking of the body 1 of the robot
    uiController.createWindow("flip_cimpc", false); // NOTE: if the saveVideo is set to true, the raw recorded file could be 2.5 GB for 15 seconds!

    // initialize variables
    int model_nv = cimpc.nv;

    mju_copy(mj_data->qpos, mj_model->key_qpos, mj_model->nq * 1); // set ini pos in Mujoco

    std::vector<double> motors_pos_des(model_nv - 6, 0);
    std::vector<double> motors_vel_des(model_nv - 6, 0);
    std::vector<double> motors_tau_des(model_nv - 6, 0);
    // std::cout << "model_nv: " << a1_model.nv << std::endl;
    for (int i = 0; i < model_nv - 6; i++)
    {
        // std::cout << "qpos[" << i << "] = " << mj_data->qpos[i] << std::endl;
        motors_pos_des[i] = mj_data->qpos[7 + i];
        // motors_pos_des[i] = cimpc.x0[i];
    }

    // register variable name for data logger
    logger.addIterm("simTime", 1);
    // logger.addIterm("motor_pos_des", model_nv - 6);
    // logger.addIterm("motor_pos_cur", model_nv - 6);
    // logger.addIterm("motor_vel_des", model_nv - 6);
    // logger.addIterm("motor_vel_cur", model_nv - 6);
    logger.addIterm("motors_tor_out", model_nv - 6);
    logger.finishItermAdding();
    //// -------------------------- main loop --------------------------------

    int CIMPC_count = 0;        // count for controlling the mpc running period
    double getup_percent = 0.0; // get up percent

    double startStandingTime = 1.5;
    double startTaskTime = 3.5;
    double startFallTime = 6;
    double simEndTime = 15;

    mjtNum simstart = mj_data->time;
    double simTime = mj_data->time;
    bool isInit = true;
    while (!glfwWindowShouldClose(uiController.window))
    {
        simstart = mj_data->time;
        while (mj_data->time - simstart < 1.0 / 60.0 && uiController.runSim)
        { // press "1" to pause and resume, "2" to step the simulation
            mj_step(mj_model, mj_data);
            simTime = mj_data->time;
            // Read the sensors:
            mj_interface.updateSensorValues();
            mj_interface.dataBusWrite(RobotState);

            // joint number: hip-l: 0-2, hip-r: 3-5, thigh-l: 6-8, thigh-r: 9-11, calf-l: 12-14, calf-r: 14-16

            if (simTime > startTaskTime)
            {
                // std::cout << "flip" << std::endl;
                RobotState.motionState = DataBus::Walk; // start walking
                // ------------- CIMPC ------------
                CIMPC_count = CIMPC_count + 1;
                if (CIMPC_count > (dt_Hz / dt - 1))
                {
                    cimpc.dataBusRead(RobotState);
                    // std::cout << "u_cur" << cimpc.u_cur.transpose() << std::endl;
                    if (isInit)
                    {
                        // cimpc.pose = "trot";
                        cimpc.cal(isInit);
                        isInit = false;
                    }
                    else
                    {
                        // cimpc.x_target = cimpc.x_init;
                        // cimpc.x_target[0] = cimpc.x_cur[0] + 0.55;
                        cimpc.cal(isInit);
                    }
                    // std::cout << "x_target: " << cimpc.x_target.transpose() << std::endl;
                    cimpc.dataBusWrite(RobotState);
                    // std::cout << "u_des" << RobotState.motors_tor_des << std::endl;
                    CIMPC_count = 0;
                }
                // get the final joint command
            }
            else if (simTime > startStandingTime)
            {
                // std::cout << "get up" << std::endl;
                RobotState.motionState = DataBus::Stand; // start standing
                getup_percent += 1 / 500.0;
                getup_percent = getup_percent > 1.0 ? 1.0 : getup_percent;
                auto interpolate = [getup_percent](double a, double b)
                {
                    return getup_percent * a + (1 - getup_percent) * b;
                };
                // get the final joint command
                // RobotState.motors_pos_des = getup_percent * motors_pos_des + (1 - getup_percent) * RobotState.motors_pos_cur;
                transform(motors_pos_des.begin(), motors_pos_des.end(),
                          RobotState.motors_pos_cur.begin(),
                          RobotState.motors_pos_des.begin(), interpolate);
                RobotState.motors_vel_des = motors_vel_des;
                RobotState.motors_tor_des = motors_tau_des;
            }

            // joint PVT controller
            pvtCtr.dataBusRead(RobotState);
            pvtCtr.dataBusWrite(RobotState);

            // give the joint torque command to Webots
            mj_interface.setMotorsTorque(RobotState.motors_tor_out);

            // data save
            logger.startNewLine();
            logger.recItermData("simTime", simTime);
            // logger.recItermData("motor_pos_des", RobotState.motors_pos_des);
            // logger.recItermData("motor_pos_cur", RobotState.motors_pos_cur);
            // logger.recItermData("motor_vel_des", RobotState.motors_vel_des);
            // logger.recItermData("motor_vel_cur", RobotState.motors_vel_cur);
            logger.recItermData("motors_tor_out", RobotState.motors_tor_out);
            logger.finishLine();
        }

        if (mj_data->time >= simEndTime)
            break;

        uiController.updateScene();
    };
    // free visualization storage
    uiController.Close();

    return 0;
}
