from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat
from casadi import cos
from casadi import sin
from fancy_plots import fancy_plots_2, fancy_plots_1

def f_system_model():
    # Name of the system
    model_name = 'Drone_ode'
    # Dynamic Values of the system
    m = 1.0  
    g = 9.81 
    Ixx = 0.02 

    # set up states & controls
    y1 = MX.sym('y1')
    z1 = MX.sym('z1')
    psi = MX.sym('psi')
    dy1 = MX.sym('dy1')
    dz1 = MX.sym('dz1')
    dpsi = MX.sym('dpsi')

    x = vertcat(y1, z1, psi, dy1, dz1, dpsi)

    F = MX.sym('F')
    T = MX.sym('T')
    u = vertcat(F, T)

    y1_dot = MX.sym('y1_dot')
    z1_dot = MX.sym('z1_dot')
    psi_dot = MX.sym('psi_dot')
    dy1_dot = MX.sym('dy1_dot')
    dz1_dot = MX.sym('dz1_dot')
    dpsi_dot = MX.sym('dpsi_dot')

    xdot = vertcat(y1_dot, z1_dot, psi_dot, dy1_dot, dz1_dot, dpsi_dot)

    R_system = MX.zeros(6, 2)
    R_system[3, 0] = (-1/m)*sin(psi)
    R_system[3, 1] = 0.0
    R_system[4, 0] = (1/m)*cos(psi)
    R_system[4, 1] = 0.0
    R_system[5, 0] = 0.0
    R_system[5, 1] = 1/Ixx

    # Internal states of the system
    h = MX.zeros(6,1)
    h[0, 0] = dy1
    h[1, 0] = dz1
    h[2, 0] = dpsi
    h[3, 0] = 0.0
    h[4, 0] = -g
    h[5, 0] = 0.0
    # dynamics
    f_expl = h + R_system@u
    f_system = Function('system',[x, u], [f_expl])

    # Acados Model
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name

    return model, f_system

def f_d(x, u, ts, f_sys):
    k1 = f_sys(x, u)
    k2 = f_sys(x+(ts/2)*k1, u)
    k3 = f_sys(x+(ts/2)*k2, u)
    k4 = f_sys(x+(ts)*k3, u)
    x = x + (ts/6)*(k1 +2*k2 +2*k3 +k4)
    aux_x = np.array(x[:,0]).reshape((6,))
    return aux_x

def create_ocp_solver_description(x0, N_horizon, t_horizon, F_max, F_min, T_max, T_min) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system = f_system_model()
    ocp.model = model
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost
    Q_mat = 1 * np.diag([1, 1, 0.0, 0.0, 0.0, 0.0])  # [x,th,dx,dth]
    R_mat = 1 * np.diag([0.0000001,  0.0000001])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    ny = nx + nu
    ny_e = nx

    ocp.cost.W_e = Q_mat
    ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.Vx = np.zeros((ny, nx))
    ocp.cost.Vx[:nx, :nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, 0:nu] = np.eye(nu)
    ocp.cost.Vu = Vu

    ocp.cost.Vx_e = np.eye(nx)

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # set constraints
    ocp.constraints.lbu = np.array([F_min, T_min])
    ocp.constraints.ubu = np.array([F_max, T_max])
    ocp.constraints.idxbu = np.array([0, 1])

    #ocp.constraints.lbx = np.array([-12])
    #ocp.constraints.ubx = np.array([12])
    #ocp.constraints.idxbx = np.array([1])
    #ocp.constraints.lbu = np.array([model.dthrottle_min, model.ddelta_min])
    #ocp.constraints.ubu = np.array([model.dthrottle_max, model.ddelta_max])
    #ocp.constraints.idxbu = np.array([0, 1])

    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 3
    ocp.solver_options.nlp_solver_max_iter = 100
    ocp.solver_options.levenberg_marquardt = 1e-2

    # set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp

def main():
    # Initial Values System
    # Simulation Time
    t_final = 20
    # Sample time
    t_s = 0.03
    # Prediction Time
    t_prediction= 2;

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]
    print(N_prediction)

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    # Parameters of the system
    g = 9.8
    m0 = 1.0
    I_xx = 0.02
    L = [g, m0, I_xx]

    # Vector Initial conditions
    x = np.zeros((6, t.shape[0]+1-N_prediction), dtype = np.double)
    x[0,0] = 0.0
    x[1,0] = 1.0
    x[2,0] = 0*(np.pi)/180
    x[3,0] = 0.0
    x[4,0] = 0.0
    x[5,0] = 0.0
    # Initial Control values
    u_control = np.zeros((2, t.shape[0]-N_prediction), dtype = np.double)

    # Reference Trajectory
    # Reference Signal of the system
    xref = np.zeros((8, t.shape[0]), dtype = np.double)
    xref[0,:] =  4 * np.sin(5*0.08*t)
    xref[1,:] = 2.5 * np.sin (0.2 * t) +5 
    xref[2,:] = 0.0
    xref[3,:] = 0.0 
    xref[4,:] = 0.0
    xref[5,:] = 0.0

    # Load the model of the system
    model, f = f_system_model()

    # Maximiun Values
    f_max = 3*m0*g
    f_min = 0*m0*g

    t_max = 0.5
    t_min = -0.5

    # Optimization Solver
    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, f_max, f_min, t_max, t_min)
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    # Init states system
    # Dimentions System
    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", 0.0 * np.ones(x[:,0].shape))
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))
    # Simulation System

    for k in range(0, t.shape[0]-N_prediction):
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # update yref
        for j in range(N_prediction):
            yref = xref[:,k+j]
            acados_ocp_solver.set(j, "yref", yref)
        yref_N = xref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "yref", yref_N[0:6])

        # Get Computational Time
        tic = time.time()
        # solve ocp
        status = acados_ocp_solver.solve()

        toc = time.time()- tic
        print(toc)

        # Get Control Signal
        u_control[:, k] = acados_ocp_solver.get(0, "u")
        # System Evolution
        x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
        delta_t[:, k] = toc
        
    fig2, ax11 = fancy_plots_1()
    ## Axis definition necesary to fancy plots
    #ax11.set_xlim((x[0, 0], x[0, -1]))
    #ax11.set_ylim((x[1, 0], x[1, -1]))

    states_1, = ax11.plot(x[0,:], x[1,:],
                    color='#00429d', lw=2, ls="-")
    states_1d, = ax11.plot(xref[0,:], xref[1,:],
                    color='#00429d', lw=2, ls="-.")

    ax11.set_ylabel(r"$[z]$", rotation='vertical')
    ax11.set_xlabel(r"$[y]$", labelpad=5)
    ax11.legend([states_1,states_1d],
            [r'$x$', r'$x_d$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax11.grid(color='#949494', linestyle='-.', linewidth=0.5)

    fig2.savefig("states.eps")
    fig2.savefig("states.png")
    fig2
    plt.show()

    fig3, ax12 = fancy_plots_1()
    ## Axis definition necesary to fancy plots
    ax12.set_xlim((t[0], t[-1]))

    control_1, = ax12.plot(t[0:u_control.shape[1]],u_control[0,:],
                    color='#00429d', lw=2, ls="-")
    control_2, = ax12.plot(t[0:u_control.shape[1]],u_control[1,:],
                    color='#9e4941', lw=2, ls="-.")

    ax12.set_ylabel(r"$[N]$", rotation='vertical')
    ax12.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
    ax12.legend([control_1, control_2],
            [r'$F$', r'$T$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax12.grid(color='#949494', linestyle='-.', linewidth=0.5)

    fig3.savefig("control_actions.eps")
    fig3.savefig("control_actions.png")
    fig3
    plt.show()

    fig4, ax13 = fancy_plots_1()
    ## Axis definition necesary to fancy plots
    ax13.set_xlim((t[0], t[-1]))

    time_1, = ax13.plot(t[0:delta_t.shape[1]],delta_t[0,:],
                    color='#00429d', lw=2, ls="-")
    tsam1, = ax13.plot(t[0:t_sample.shape[1]],t_sample[0,:],
                    color='#9e4941', lw=2, ls="-.")

    ax13.set_ylabel(r"$[s]$", rotation='vertical')
    ax13.set_xlabel(r"$\textrm{Time}[s]$", labelpad=5)
    ax13.legend([time_1,tsam1],
            [r'$t_{compute}$',r'$t_{sample}$'],
            loc="best",
            frameon=True, fancybox=True, shadow=False, ncol=2,
            borderpad=0.5, labelspacing=0.5, handlelength=3, handletextpad=0.1,
            borderaxespad=0.3, columnspacing=2)
    ax13.grid(color='#949494', linestyle='-.', linewidth=0.5)

    fig4.savefig("time.eps")
    fig4.savefig("time.png")
    fig4
    plt.show()
    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')

        
if __name__ == '__main__':
    main()
