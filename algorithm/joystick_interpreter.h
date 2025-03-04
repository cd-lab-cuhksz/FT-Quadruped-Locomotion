#pragma once

#include "task_trajectory.h"
#include "data_bus.h"

class JoyStickInterpreter
{
public:
    double dt;        // control frequency
    double thetaZ{0}; // yaw angle command in body frame
    JoyStickInterpreter(double dtIn) : dt{dtIn}, vxLGen(dtIn), vyLGen(dtIn), wzLGen(dtIn) {};
    void setIniPos(double posX, double posY, double thetaZ);
    void step();
    double vx_W{0}, vy_W{0};          // generated velocity in x and y direction w.r.t world frame
    double px_W{0}, py_W{0};          // generated position in x and y direction w.r.t world frame
    double vx_L{0}, vy_L{0}, wz_L{0}; // generated linear velocity in x and y direction, angular velocity in z direction, w.r.t body frame
    void dataBusWrite(DataBus &dataBus);
    void reset();
    TaskTrajectory vxLGen, vyLGen, wzLGen;
    // private:
};
