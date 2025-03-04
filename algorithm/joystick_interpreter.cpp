#include "joystick_interpreter.h"

void JoyStickInterpreter::step()
{
    vx_L = vxLGen.step();
    vy_L = vyLGen.step();
    wz_L = wzLGen.step();

    thetaZ = thetaZ + wz_L * dt;

    vx_W = cos(thetaZ) * vx_L - sin(thetaZ) * vy_L;
    vy_W = sin(thetaZ) * vx_L + cos(thetaZ) * vy_L;

    px_W += vx_W * dt;
    py_W += vy_W * dt;
}

// NOTE: currently only the  x, y directions are controlled. Walking on a slope is not considered here.
void JoyStickInterpreter::dataBusWrite(DataBus &dataBus)
{
    dataBus.js_pos_des[0] = px_W;
    dataBus.js_pos_des[1] = py_W;
    dataBus.js_vel_des[0] = vx_W;
    dataBus.js_vel_des[1] = vy_W;
    dataBus.js_eul_des[2] = thetaZ;
    dataBus.js_omega_des[2] = wz_L;
}

void JoyStickInterpreter::reset()
{
    vxLGen.resetOut(0);
    vyLGen.resetOut(0);
    wzLGen.resetOut(0);
    vx_L = 0;
    vy_L = 0;
    wz_L = 0;
    thetaZ = 0;
}

void JoyStickInterpreter::setIniPos(double posX, double posY, double thetaZ)
{
    px_W = posX;
    py_W = posY;
    this->thetaZ = thetaZ;
}
