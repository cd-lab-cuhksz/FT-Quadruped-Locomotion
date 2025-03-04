#pragma once

#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include "json/json.h"
#include "data_bus.h"
#include "utils.h"
#include <pinocchio/algorithm/model.hpp>
#include <pinocchio/parsers/srdf.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include "crocoddyl/core/activations/weighted-quadratic.hpp"
#include "crocoddyl/multibody/residuals/state.hpp"
#include "crocoddyl/multibody/residuals/frame-velocity.hpp"
#include "crocoddyl/multibody/residuals/frame-translation.hpp"
#include "crocoddyl/core/activations/quadratic-barrier.hpp"
#include "crocoddyl/core/solvers/box-fddp.hpp"

//  For Pinocchio: The base translation part is expressed in the parent frame (here the world coordinate system)
//  while its velocity is expressed in the body coordinate system.
//  https://github.com/stack-of-tasks/pinocchio/issues/1137
//  q = [global_base_position, global_base_quaternion, joint_positions]
//  v = [local_base_velocity_linear, local_base_velocity_angular, joint_velocities]

class CIMPC
{
public:
    Eigen::VectorXd x_cur, u_cur, x_target, x_init;
    std::string pose = "bounding";
    int nv;

    CIMPC(double dtIn, std::string urdf_pathIn, std::string srdf_pathIn);
    void dataBusRead(DataBus &Data);
    void dataBusWrite(DataBus &Data);

    // cost function
    std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> handstand(const pinocchio::Model &rmodel, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, const Eigen::VectorXd &x0);

    std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> upright(const pinocchio::Model &rmodel, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, const Eigen::VectorXd &x0);
    std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> crouch(const pinocchio::Model &rmodel, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, const Eigen::VectorXd &x0);
    void add_lf_cost(boost::shared_ptr<crocoddyl::CostModelSum> costs, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, std::vector<int> contact_ids);
    void set_lf_cost(boost::shared_ptr<crocoddyl::CostModelSum> costs, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, int contact_id);
    void set_la_cost(boost::shared_ptr<crocoddyl::CostModelSum> costs, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, int contact_id, bool active = true);
    // cost model for general one shoot problem
    std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> target_cost(const pinocchio::Model &rmodel, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, const Eigen::VectorXd &target, const std::string &pose = "bounding");

    void cal(bool is_init);

private:
    // simulate parameters
    double dt;
    pinocchio::Model model;
    boost::shared_ptr<crocoddyl::StateMultibody> state;
    boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation;
    boost::shared_ptr<ContactModel> contact_model;

    std::vector<std::string> foot_names = {"FL_foot", "FR_foot", "RL_foot", "RR_foot"};
    std::vector<std::string> calf_names = {"FL_calf", "FR_calf", "RL_calf", "RR_calf"};
    std::string body = "trunk";

    std::vector<int> contact_ids;
    std::vector<double> friction = {0.8, 0.8, 0.8, 0.8};

    std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> costs;
    std::vector<boost::shared_ptr<crocoddyl::ActionModelAbstract>> actionmodels;

    // solver parameters
    int N = 19; // 20-1
    double rho = 4;
    int maxiter = 4;
    // bool is_feasible = false;
    bool is_feasible = false;
    double init_reg = 0.1;
    std::vector<Eigen::VectorXd> xs;
    std::vector<Eigen::VectorXd> us;
    Eigen::VectorXd x_des, u_des;

    // cost weights
    int alpha = 30;
    int beta = 10;

    double wf = 1;
    double wa = 2e3;
    double ws = 1e-1;
    double w_bound = 1e6;
    double height;

    Eigen::MatrixXd C_trot, C_pace, C_bounding;
    Eigen::VectorXd x_lb, x_ub;
    Eigen::VectorXd wf_act, wa_act;

    boost::shared_ptr<crocoddyl::ActivationModelWeightedQuad> x_act_r, x_act_t;
    boost::shared_ptr<crocoddyl::ActivationModelQuad> ls_act;
};