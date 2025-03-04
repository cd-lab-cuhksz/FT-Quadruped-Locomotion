#include "cimpc.h"

CIMPC::CIMPC(double dtIn, std::string urdf_pathIn, std::string srdf_pathIn)
{
    dt = dtIn;
    pinocchio::urdf::buildModel(urdf_pathIn, pinocchio::JointModelFreeFlyer(), model);
    pinocchio::srdf::loadReferenceConfigurations(model, srdf_pathIn, false);
    nv = model.nv;

    auto lims = model.effortLimit;
    lims *= 0.5; // reduced artificially the torque limits
    model.effortLimit = lims;

    state = boost::make_shared<crocoddyl::StateMultibody>(boost::make_shared<pinocchio::Model>(model));
    actuation = boost::make_shared<crocoddyl::ActuationModelFloatingBase>(state);

    int lf_id = model.getFrameId(foot_names[0]);
    int rf_id = model.getFrameId(foot_names[1]);
    int lh_id = model.getFrameId(foot_names[2]);
    int rh_id = model.getFrameId(foot_names[3]);
    int body_id = model.getFrameId(body);
    int lfcalf_id = model.getFrameId(calf_names[0]);
    int rfcalf_id = model.getFrameId(calf_names[1]);
    int lhcalf_id = model.getFrameId(calf_names[2]);
    int rhcalf_id = model.getFrameId(calf_names[3]);

    contact_ids = {lf_id, rf_id, lh_id, rh_id};
    contact_model = boost::make_shared<ContactModel>(state, contact_ids, friction);

    auto q0 = model.referenceConfigurations["standing"];                                       // 19
    x_init = npConcatenate(std::vector<Eigen::VectorXd>{q0, Eigen::VectorXd::Zero(model.nv)}); // 19 + 18
    u_cur = Eigen::VectorXd::Zero(actuation->get_nu());
    x_cur = x_init;
    x_target = x_init;
    xs = std::vector<Eigen::VectorXd>(N + 1, x_init); // 19 + 1
    us = std::vector<Eigen::VectorXd>(N, u_cur);

    auto tmp_nv = state->get_nv();
    auto tmp_lb = state->get_lb();
    auto tmp_ub = state->get_ub();
    x_lb = npConcatenate(std::vector<Eigen::VectorXd>{tmp_lb.segment(1, tmp_nv), tmp_lb.bottomRows(tmp_nv)});
    x_ub = npConcatenate(std::vector<Eigen::VectorXd>{tmp_ub.segment(1, tmp_nv), tmp_ub.bottomRows(tmp_nv)});

    Eigen::MatrixXd D(2, 3);
    D << 0, 1, 0,
        0, 0, 1;
    Eigen::MatrixXd c_trot(2, 4);
    c_trot << 1, 0, 0, -1,
        0, 1, -1, 0;
    Eigen::MatrixXd c_pace(2, 4);
    c_pace << 1, 0, -1, 0,
        0, 1, 0, -1;
    Eigen::MatrixXd c_bounding(2, 4);
    c_bounding << 1, -1, 0, 0,
        0, 0, 1, -1;
    C_trot = Eigen::kroneckerProduct(c_trot, D);
    C_pace = Eigen::kroneckerProduct(c_pace, D);
    C_bounding = Eigen::kroneckerProduct(c_bounding, D);
    // std::cout << C_trot << std::endl;
    // std::cout << C_pace << std::endl;
    // std::cout << C_bounding << std::endl;

    wf_act = Eigen::VectorXd::Zero(6);
    wa_act = Eigen::VectorXd::Zero(3);
    wf_act << 1, 1, 0, 0, 0, 0;
    wa_act << 0, 0, 1;

    Eigen::VectorXd wq(18);
    wq << 20, 20, 80, 10, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    Eigen::VectorXd wv = wq / static_cast<double>(alpha);
    Eigen::VectorXd wx_r = npConcatenate(std::vector<Eigen::VectorXd>{wq, wv});
    Eigen::VectorXd wx_t = static_cast<double>(beta) * wx_r;
    x_act_r = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(2 * (wx_r.array().pow(1.7)));
    x_act_t = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(2 * (wx_t.array().pow(1.7)));
    ls_act = boost::make_shared<crocoddyl::ActivationModelQuad>(4);

    // read cimpc parameters
    Json::Reader reader;
    Json::Value root_read;
    std::ifstream in("../config/cimpc_config.json", std::ios::binary);

    reader.parse(in, root_read);
    N = root_read["N"].asInt();
    rho = root_read["rho"].asDouble();
    maxiter = root_read["maxiter"].asInt();
    init_reg = root_read["init_reg"].asDouble();
}

void CIMPC::dataBusRead(DataBus &Data)
{
    x_cur.segment(0, 19) = Data.q;
    x_cur.segment(19, 18) = Data.dq;
    for (int i = 0; i < 12; i++)
    {
        u_cur(i) = Data.motors_tor_cur[i];
    }
}

void CIMPC::dataBusWrite(DataBus &Data)
{
    Data.motors_pos_des = eigen2std(x_des.segment(7, 12));
    Data.motors_vel_des = eigen2std(x_des.segment(25, 12));
    Data.motors_tor_des = eigen2std(u_des);
}

std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> CIMPC::handstand(const pinocchio::Model &rmodel, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, const Eigen::VectorXd &x0)
{
    boost::shared_ptr<crocoddyl::CostModelSum> lr = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
    boost::shared_ptr<crocoddyl::CostModelSum> lt = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

    Eigen::VectorXd xtarget = x0;
    // xtarget << 0, 0, 0.42, 0, 0, 0, 1, 0.1000, 0.8000, -1.5000, -0.1000, 0.8000, -1.5000, 0.1000, 1.0000, -1.5000, -0.1000, 1.0000, -1.5000;
    auto quat = rpyToQuat(Eigen::Vector3d(0.0, 0.9, 0));
    xtarget(2) += 0.1;
    xtarget(3) = quat.x();
    xtarget(4) = quat.y();
    xtarget(5) = quat.z();
    xtarget(6) = quat.w();
    // std::cout << x_act_r->get_nr() << std::endl;
    auto tmp = boost::make_shared<crocoddyl::ResidualModelState>(state, xtarget, actuation->get_nu());
    // std::cout << tmp->get_nr() << std::endl;
    auto lr_x = boost::make_shared<crocoddyl::CostModelResidual>(state, x_act_r, boost::make_shared<crocoddyl::ResidualModelState>(state, xtarget, actuation->get_nu()));
    auto lt_x = boost::make_shared<crocoddyl::CostModelResidual>(state, x_act_t, boost::make_shared<crocoddyl::ResidualModelState>(state, xtarget, actuation->get_nu()));
    // auto lr_u = boost::make_shared<crocoddyl::CostModelResidual>(state, boost::make_shared<crocoddyl::ResidualModelState>(state, actuation->get_nu()));
    auto ls = boost::make_shared<ControlSymmCostModel>(state, ls_act, C_bounding, actuation->get_nu());

    lr->addCost("xGoal", lr_x, 2);
    lr->addCost("ls", ls, 2 * ws);
    lt->addCost("xGoal", lt_x, 2);

    return {lr, lt};
}

std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> CIMPC::upright(const pinocchio::Model &rmodel, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, const Eigen::VectorXd &x0)
{
    boost::shared_ptr<crocoddyl::CostModelSum> lr = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
    boost::shared_ptr<crocoddyl::CostModelSum> lt = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

    Eigen::VectorXd xtarget = x0;
    // xtarget << 0, 0, 0.42, 0, 0, 0, 1, 0.1000, 0.8000, -1.5000, -0.1000, 0.8000, -1.5000, 0.1000, 1.0000, -1.5000, -0.1000, 1.0000, -1.5000;
    auto quat = rpyToQuat(Eigen::Vector3d(0.0, -3.14 / 2, 0));
    xtarget(2) += 0.3;
    xtarget(3) = quat.x();
    xtarget(4) = quat.y();
    xtarget(5) = quat.z();
    xtarget(6) = quat.w();
    xtarget(8) = 3.14 / 2;
    xtarget(11) = 3.14 / 2;
    xtarget(14) = 3.14 / 2;
    xtarget(15) = 0;
    xtarget(17) = 3.14 / 2;
    xtarget(18) = 0;

    auto lr_x = boost::make_shared<crocoddyl::CostModelResidual>(state, x_act_r, boost::make_shared<crocoddyl::ResidualModelState>(state, xtarget, actuation->get_nu()));
    auto lt_x = boost::make_shared<crocoddyl::CostModelResidual>(state, x_act_t, boost::make_shared<crocoddyl::ResidualModelState>(state, xtarget, actuation->get_nu()));
    // auto lr_u = boost::make_shared<crocoddyl::CostModelResidual>(state, boost::make_shared<crocoddyl::ResidualModelState>(state, actuation->get_nu()));
    auto ls = boost::make_shared<ControlSymmCostModel>(state, ls_act, C_bounding, actuation->get_nu());

    lr->addCost("xGoal", lr_x, 2);
    lr->addCost("ls", ls, 2 * ws);
    lt->addCost("xGoal", lt_x, 2);

    return {lr, lt};
}

std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> CIMPC::crouch(const pinocchio::Model &rmodel, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, const Eigen::VectorXd &x0)
{
    boost::shared_ptr<crocoddyl::CostModelSum> lr = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
    boost::shared_ptr<crocoddyl::CostModelSum> lt = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

    Eigen::VectorXd xtarget = x0;
    // xtarget << 0, 0, 0.42, 0, 0, 0, 1, 0.1000, 0.8000, -1.5000, -0.1000, 0.8000, -1.5000, 0.1000, 1.0000, -1.5000, -0.1000, 1.0000, -1.5000;
    auto quat = rpyToQuat(Eigen::Vector3d(0.0, 0, 0));
    xtarget(2) -= 0.1;
    xtarget(3) = quat.x();
    xtarget(4) = quat.y();
    xtarget(5) = quat.z();
    xtarget(6) = quat.w();

    auto lr_x = boost::make_shared<crocoddyl::CostModelResidual>(state, x_act_r, boost::make_shared<crocoddyl::ResidualModelState>(state, xtarget, actuation->get_nu()));
    auto lt_x = boost::make_shared<crocoddyl::CostModelResidual>(state, x_act_t, boost::make_shared<crocoddyl::ResidualModelState>(state, xtarget, actuation->get_nu()));
    // auto lr_u = boost::make_shared<crocoddyl::CostModelResidual>(state, boost::make_shared<crocoddyl::ResidualModelState>(state, actuation->get_nu()));
    auto ls = boost::make_shared<ControlSymmCostModel>(state, ls_act, C_bounding, actuation->get_nu());

    // auto stateBoundsResidual = boost::make_shared<crocoddyl::ResidualModelState>(state, actuation->get_nu());
    // auto stateBoundsActivation = boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(crocoddyl::ActivationBounds(x_lb, x_ub));
    // auto stateBounds = boost::make_shared<crocoddyl::CostModelResidual>(state, stateBoundsActivation, stateBoundsResidual);

    lr->addCost("xGoal", lr_x, 2);
    lr->addCost("ls", ls, 2 * ws);
    lt->addCost("xGoal", lt_x, 2);

    return {lr, lt};
}

void CIMPC::add_lf_cost(boost::shared_ptr<crocoddyl::CostModelSum> costs, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, std::vector<int> contact_ids)
{
    for (auto i : contact_ids)
    {
        auto lf_res = boost::make_shared<crocoddyl::ResidualModelFrameVelocity>(state, i, pinocchio::Motion::Zero(), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, actuation->get_nu());
        auto lf_act = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(wf_act);
        auto lf = boost::make_shared<crocoddyl::CostModelResidual>(state, lf_act, lf_res);
        auto costname = "lf_" + std::to_string(i);
        if (costs->get_active_set().count(costname) == 0 && costs->get_inactive_set().count(costname) == 0)
        {
            costs->addCost(costname, lf, 2 * wf);
        }
        else
        {
            costs->removeCost(costname);
            costs->addCost(costname, lf, 2 * wf);
        }
    }
}

void CIMPC::set_lf_cost(boost::shared_ptr<crocoddyl::CostModelSum> costs, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, int contact_id)
{
    auto costname = "lf_" + std::to_string(contact_id);
    if (costs->get_active_set().count(costname) == 0 && costs->get_inactive_set().count(costname) == 0)
    {
        return;
    }

    auto wf_act_height = wf_act * 0.5 * (1 + std::tanh(-15 * height));
    auto lf_res = boost::make_shared<crocoddyl::ResidualModelFrameVelocity>(state, contact_id, pinocchio::Motion::Zero(), pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, actuation->get_nu());
    auto lf_act = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(wf_act_height);
    auto lf = boost::make_shared<crocoddyl::CostModelResidual>(state, lf_act, lf_res);
    costs->removeCost(costname);
    costs->addCost(costname, lf, 2 * wf);
}

void CIMPC::set_la_cost(boost::shared_ptr<crocoddyl::CostModelSum> costs, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, int contact_id, bool active)
{
    auto costname = "lf_" + std::to_string(contact_id);
    if (!active)
    {
        if (costs->get_active_set().count(costname) == 0 && costs->get_inactive_set().count(costname) == 0)
        {
            costs->removeCost(costname);
        }
        return;
    }

    auto la_res = boost::make_shared<crocoddyl::ResidualModelFrameTranslation>(state, contact_id, Eigen::VectorXd::Zero(3), actuation->get_nu());
    auto la_act = boost::make_shared<crocoddyl::ActivationModelWeightedQuad>(wa_act.array().square());
    auto la = boost::make_shared<crocoddyl::CostModelResidual>(state, la_act, la_res);

    if (costs->get_active_set().count(costname) == 0 && costs->get_inactive_set().count(costname) == 0)
    {
        costs->addCost(costname, la, 2 * wa);
    }
    else
    {
        costs->removeCost(costname);
        costs->addCost(costname, la, 2 * wa);
    }
}

std::vector<boost::shared_ptr<crocoddyl::CostModelSum>> CIMPC::target_cost(const pinocchio::Model &rmodel, boost::shared_ptr<crocoddyl::StateMultibody> state, boost::shared_ptr<crocoddyl::ActuationModelFloatingBase> actuation, const Eigen::VectorXd &target, const std::string &pose)
{
    boost::shared_ptr<crocoddyl::CostModelSum> lr = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());
    boost::shared_ptr<crocoddyl::CostModelSum> lt = boost::make_shared<crocoddyl::CostModelSum>(state, actuation->get_nu());

    // Eigen::VectorXd xtarget = x0;
    Eigen::VectorXd xtarget = target;
    // xtarget << 0, 0, 0.42, 0, 0, 0, 1, 0.1000, 0.8000, -1.5000, -0.1000, 0.8000, -1.5000, 0.1000, 1.0000, -1.5000, -0.1000, 1.0000, -1.5000;

    auto lr_x = boost::make_shared<crocoddyl::CostModelResidual>(state, x_act_r, boost::make_shared<crocoddyl::ResidualModelState>(state, xtarget, actuation->get_nu()));
    auto lt_x = boost::make_shared<crocoddyl::CostModelResidual>(state, x_act_t, boost::make_shared<crocoddyl::ResidualModelState>(state, xtarget, actuation->get_nu()));

    auto stateBoundsResidual = boost::make_shared<crocoddyl::ResidualModelState>(state, actuation->get_nu());
    auto stateBoundsActivation = boost::make_shared<crocoddyl::ActivationModelQuadraticBarrier>(crocoddyl::ActivationBounds(x_lb, x_ub));
    auto stateBounds = boost::make_shared<crocoddyl::CostModelResidual>(state, stateBoundsActivation, stateBoundsResidual);

    lr->addCost("xGoal", lr_x, 2);

    if (pose != "")
    {
        boost::shared_ptr<ControlSymmCostModel> ls;
        if (pose == "bounding")
        {
            ls = boost::make_shared<ControlSymmCostModel>(state, ls_act, C_bounding, actuation->get_nu());
        }
        else if (pose == "trot")
        {
            ls = boost::make_shared<ControlSymmCostModel>(state, ls_act, C_trot, actuation->get_nu());
        }
        else if (pose == "pace")
        {
            ls = boost::make_shared<ControlSymmCostModel>(state, ls_act, C_pace, actuation->get_nu());
        }
        lr->addCost("ls", ls, 2 * ws);
    }
    lr->addCost("lx_bound", stateBounds, w_bound);
    lt->addCost("xGoal", lt_x, 2);

    return {lr, lt};
}

void CIMPC::cal(bool is_init)
{
    if (is_init)
    {
        x_init = x_cur;
        x_init.segment(19, 18) = Eigen::VectorXd::Zero(18);
        x_target = x_init;
        xs = std::vector<Eigen::VectorXd>(N + 1, x_init); // 19 + 1
        // us = std::vector<Eigen::VectorXd>(N, u_cur);
        us = std::vector<Eigen::VectorXd>(N, Eigen::VectorXd::Zero(12));
    }
    costs = target_cost(model, state, actuation, x_target, pose);
    add_lf_cost(costs[0], state, actuation, contact_ids);
    // for (auto i : contact_ids)
    // {
    //     set_lf_cost(costs[0], state, actuation, i);
    //     set_la_cost(costs[0], state, actuation, i, true);
    // }
    actionmodels = IAM_shoot(N, state, actuation, costs, contact_model, dt, rho);
    auto actionmodel_1 = actionmodels.back();
    actionmodels.pop_back();
    boost::shared_ptr<crocoddyl::ShootingProblem> problem =
        boost::make_shared<crocoddyl::ShootingProblem>(x_cur, actionmodels, actionmodel_1);
    crocoddyl::SolverBoxFDDP solver(problem);
    solver.solve(xs, us, maxiter, is_feasible, init_reg);
    xs = solver.get_xs();
    us = solver.get_us();
    x_des = xs[0];
    u_des = us[0];
    xs.erase(xs.begin());
    us.erase(us.begin());
    xs.push_back(xs.back());
    us.push_back(Eigen::VectorXd::Zero(12));
    // xs.push_back(pinocchio::integrate(model, xs.back(), us.back() * dt));
}