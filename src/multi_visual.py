import numpy as np
import casadi
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib as mpl
from mpc_cbf.plan_dubins import plan_dubins_path
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from utils import dm_to_array, align_length
from env import GridWorld
from matplotlib.lines import Line2D
import torch

mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True

def simulate(world, n_agents, cat_states_list, heatmaps, cov_lvls, obstacles, num_frames, init_list, save, save_path, cost_values=None, reward_values=None):
    """
    Simulate and visualize the multi-agent system with real-time cost and reward metrics.
    
    Args:
        world: The environment object
        n_agents: Number of agents in the simulation
        cat_states_list: List of agent state trajectories
        heatmaps: List of heatmaps for each frame
        cov_lvls: List of coverage levels for each frame
        obstacles: List of obstacles (x, y, radius)
        num_frames: Number of frames to simulate
        init_list: Initial states of agents
        save: Boolean flag to save the animation
        save_path: Path to save the animation
        cost_values: List of cost values for each frame (optional)
        reward_values: List of reward values for each frame (optional)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.lines import Line2D
    import matplotlib.animation as animation
    
    def plot_heatmap(world, i):
        '''
        Inputs:
        world.heatmap: 2-D matrix with size (w, h). Heatmap.
        obstacle: Binary 2-D matrix with size (w, h).
        world.agents: A list contains all the agents of Agent type. Each one has 2-D position

        return:
        hm_show: RGB map containing heatmap, obstacles and agents.
        '''
        # Normalize heatmap
        hm_normed = (heatmaps[i] / world.heat_max * 255).astype(np.uint8)  # substitute world.temp_max with 0.1 will be more apparent
        hm_show = hm_normed
        return hm_show
    
    def plot_cov_lvl(world, i):
        '''
        Inputs:
        world.heatmap: 2-D matrix with size (w, h). Heatmap.
        obstacle: Binary 2-D matrix with size (w, h).
        world.agents: A list contains all the agents of Agent type. Each one has 2-D position

        return:
        hm_show: RGB map containing heatmap, obstacles and agents.
        '''
        # Normalize heatmap
        cl_normed = (cov_lvls[i] / world.cov_max * 255).astype(np.uint8)  # substitute world.temp_max with 0.1 will be more apparent
        cl_show = cl_normed
        return cl_show
    

    def create_triangle(state=[0,0,0], h=2, w=1.5, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th),  np.cos(th)]
        ])

        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    def init():
        # Initialize text elements
        cost_text.set_text("")
        reward_text.set_text("")
        return path_list + horizon_list + [cost_text, reward_text]

    def animate(i):
        for k in range(n_agents):
            # get variables
            x = cat_states_list[k][0, 0, i]
            y = cat_states_list[k][1, 0, i]
            th = cat_states_list[k][2, 0, i]

            # update horizon
            x_new = cat_states_list[k][0, :, i]
            y_new = cat_states_list[k][1, :, i]
            horizon_list[k].set_data(x_new, y_new)

            # update current_state
            current_state_list[k].set_xy(create_triangle([x, y, th], update=True))
        
        # update heatmap
        img_hm = plot_heatmap(world, i)
        hm.set_data(img_hm)

        img_cl = plot_cov_lvl(world, i)
        cl.set_data(img_cl)
        
        # Update cost and reward text 

        # if cost_values is not None and i < len(cost_values):
        #     cost_text.set_text(f"Cost: {cost_values[i]:.4f}")
        # if reward_values is not None and i < len(reward_values):
        #     reward_text.set_text(f"Reward: {reward_values[i]:.4f}")

        return horizon_list + [cost_text, reward_text]

    # create figure and axes
    fig, ax = plt.subplots(1, 2, figsize=(6, 6))
    fig.set_size_inches(19.2, 10.8)
    size_world = world.heatmap.shape
    min_scale = 0
    ax[0].set_xlim(left = min_scale, right = world.len_grid * size_world[1])
    ax[0].set_ylim(bottom = world.len_grid * size_world[0], top = min_scale)
    ax[1].set_xlim(left = min_scale, right = world.len_grid * size_world[1])
    ax[1].set_ylim(bottom = world.len_grid * size_world[0], top = min_scale)

    # Add text for cost and reward at the top of the figure
    cost_text = fig.text(0.25, 0.95, "", fontsize=14, color='red', 
                         bbox=dict(facecolor='white', alpha=0.8))
    reward_text = fig.text(0.75, 0.95, "", fontsize=14, color='green',
                          bbox=dict(facecolor='white', alpha=0.8))

    # Obstacles
    for (ox, oy, obsr) in obstacles:
        circle = plt.Circle((ox, oy), obsr, color='r')
        ax[0].add_patch(circle)

        circle1 = plt.Circle((ox, oy), obsr, color='r')
        ax[1].add_patch(circle1)

    # create lines:
    #   path
    path_list = []
    ref_path_list = []
    horizon_list = []
    current_state_list = []
    for k in range(n_agents):
        path, = ax[0].plot([], [], 'r', linewidth=2)
        ref_path, = ax[0].plot([], [], 'b', linewidth=2)
        horizon, = ax[0].plot([], [], 'x-g', alpha=0.5)
        current_triangle = create_triangle(init_list[k, :])
        current_state = ax[0].fill(current_triangle[:, 0], current_triangle[:, 1], color='y')
        current_state = current_state[0]

        path_list.append(path)
        ref_path_list.append(ref_path)
        horizon_list.append(horizon)
        current_state_list.append(current_state)


    hm = ax[0].imshow(np.ones(heatmaps[0].shape)*255, origin='lower', extent=[0., world.len_grid * size_world[0], 0, world.len_grid * size_world[1]], cmap='viridis', vmin=0, vmax=255)
    cl = ax[1].imshow(np.ones(cov_lvls[0].shape)*255, origin='lower', extent=[0., world.len_grid * size_world[0], 0, world.len_grid * size_world[1]], cmap='viridis', vmin=0, vmax=255)
    ax[0].set_xlabel('x position')
    ax[1].set_xlabel('y position')
    blue_cmp = plt.get_cmap('viridis', 256)
    cmp = plt.get_cmap('viridis', 256)
    
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=blue_cmp),
             ax=ax[0], orientation='vertical',fraction=0.046, pad=0.04, label='Importance density')
    
    fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, 1), cmap=cmp),
             ax=ax[1], orientation='vertical',fraction=0.046, pad=0.04, label='Coverage level')
    
    legend_elements = [Line2D([0], [0], marker='>', color='y', markerfacecolor='y', markersize=15, label='Robots'),
                       Line2D([0], [0], marker='o',color='r', markerfacecolor='r', markersize=15,label='Obstacles',),
                   Line2D([0], [0], marker='x',color='g', markerfacecolor='g', markersize=15,label='MPC Predicted Path',),
                   ]

    ax[0].legend(handles=legend_elements, loc='upper right')

    sim = animation.FuncAnimation(
        fig=fig,
        func = animate,
        init_func=init,
        frames=num_frames,
        interval=0.1,
        blit=False,
        repeat=False
    )
    if save == True:
        sim.save(save_path, writer='ffmpeg', fps=50)
    plt.show()
    return sim

def main(args=None):
    size_world = (50, 50)
    # rate = np.ones(size_world)*0.01
    len_grid = 1
    heatmap = np.ones(size_world) * 0.1
    heatmap[30:40, 20:30] = 0.7
    world = GridWorld(size_world, len_grid, heatmap, obstacles=None)

    Q_x = 10
    Q_y = 10
    Q_theta = 10
    R_v = 0.5
    R_omega = 0.005

    dt = 0.1
    N = 20

    r = 3 
    v = 1

    v_lim = [-2, 2]
    omega_lim = [-casadi.pi/4, casadi.pi/4]
    Q = [Q_x, Q_y, Q_theta]
    R = [R_v, R_omega]
    obstacles = [(14,4,3), (18,15,3), (6,19,3), (29, 44,3), (38,15,3), (36,29,3), (25, 26,3)]

    # TODO Move the definition of obstacles to env, not in agents
    # init_states = np.array([[0, 0, 0], [30, 20, 0]]) # [x, y, theta]
    n_agents = 3
    n_targets = 2
    # xy_goals = np.random.rand(n_agents, n_targets, 2)
    xy_goals = np.array([[[5, 45], [20, 10]], [[10, 10], [45, 5]], [[45, 30], [15, 45]]], dtype=np.float32) / 50
    # xy_goals = np.array([[[5, 45], [20, 10]], [[10, 10], [45, 5]], [[45, 45], [15, 45]]], dtype=np.float32)
    xy_goals[:, :, 0] = xy_goals[:, :, 0] * size_world[0] * len_grid
    xy_goals[:, :, 1] = xy_goals[:, :, 1] * size_world[1] * len_grid
    theta_goals = np.random.rand(n_agents, n_targets, 1)
    state_goal_list = np.dstack((xy_goals, theta_goals)) # All goal points, state_goal_list[i, 0, :] is the init position for agent i
    # state_goal_list = np.array([[40, 25, np.pi/2], [5, 40, np.pi/2]])
    t0_list = [0 for i in range(n_agents)]
    agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_goal_list[i, 0, :], obstacles= obstacles, flag_cbf=True) for i in range(n_agents)]
    ref_states_list = []
    print('Generating ref trajectories...')
    for k in range(n_targets-1):
        for i in range(n_agents):
            path_x, path_y, path_yaw, _, _ = plan_dubins_path(state_goal_list[i, k, 0], state_goal_list[i, k, 1], state_goal_list[i, k, 2],
                                                            state_goal_list[i, k+1, 0], state_goal_list[i, k+1, 1], state_goal_list[i, k+1, 2], r, step_size=v*dt)
            if k == 0:
                ref_states_list.append(np.array([path_x, path_y, path_yaw]).T)
            else:
                ref_states_list[i] = np.concatenate((ref_states_list[i], np.array([path_x, path_y, path_yaw]).T), axis=0)

    world.add_agents(agents)
    state_0_list = [casadi.DM(state_goal_list[i, 0, :]) for i in range(n_agents)]
    u0_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
    X0_list = [casadi.repmat(state_0_list[i], 1, N + 1) for i in range(n_agents)]
    cat_states_list = [dm_to_array(X0_list[i]) for i in range(n_agents)]
    cat_controls_list = [dm_to_array(u0_list[i][:, 0]) for i in range(n_agents)]
    cost_world_list, log_probs = [], []
    heatmaps = [np.copy(world.heatmap)]
    cov_lvls = [np.copy(world.cov_lvl)]
    trip_lens = [len(ref_states) for ref_states in ref_states_list]
    longest_trip = np.max(trip_lens)
    print('Computing MPC trajectories...')
    lb = [-casadi.inf, -casadi.inf, -casadi.inf]
    ub = [casadi.inf, casadi.inf, casadi.inf]
    #####################
    observations = world.check()
    dists = world.agents_dist()
    for i in range(n_agents):
        agents[i].update_neighbors(dists[i, :])
        agents[i].embed_local(torch.tensor(observations[i], dtype=torch.float32))

    for i in range(longest_trip):
        for j in range(n_agents):
            if i < len(ref_states_list[j]):
                neighbor_embed = [agents[j].local_embed for j in agents[i].neighbors]    
                actions, log_prob = agents[i].generate_waypoints(observations[i], torch.cat(neighbor_embed, dim=0))
                log_probs.append(log_prob)
                u, X_pred = agents[j].solve(X0_list[j], u0_list[j], ref_states_list[j], i, ub, lb)
            
                cat_states_list[j] = np.dstack((cat_states_list[j], dm_to_array(X_pred)))
                cat_controls_list[j] = np.dstack((cat_controls_list[j], dm_to_array(u[:, 0])))
                
                t0_list[j], X0_list[j], u0_list[j] = agents[j].shift_timestep(dt, t0_list[j], X_pred, u)
                agents[j].states = X0_list[j][:, 1]
       
        world.step()   
        # _ =world.check()     
        heatmaps.append(np.copy(world.heatmap))
        cov_lvls.append(np.copy(world.cov_lvl))
        cost_world_list.append(torch.tensor(world.get_cost_mean()))  # -np.mean(self.heatmap * self.cov_lvl)
        # cost_world_list.append(torch.tensor(world.get_cost_max()))
        heatmaps.append(np.copy(world.heatmap))
        cov_lvls.append(np.copy(world.cov_lvl))

    cost_agents = torch.zeros((n_agents,), device='cpu') # No costs of individual agent now
    cost_world = torch.sum(torch.tensor(cost_world_list).to('cpu'))
    cost = cost_agents + cost_world
    # loss = torch.matmul(torch.stack(log_probs), cost)
    loss = torch.stack(log_probs).sum() * cost.sum()

    # grad_norm = compute_gradient_norm(decisionNN)

    print("Cost world:%.3f"%cost_world.detach().cpu().numpy())
    print('loss: %.3f'%loss.detach().cpu().numpy())
    # print('Gradient norm: %.3f'%grad_norm)
    
    align_length(cat_states_list, longest_trip+1)
    align_length(cat_controls_list, longest_trip+1)
    print('Drawing...')
    simulate(world, n_agents, cat_states_list, heatmaps, cov_lvls,
              obstacles, longest_trip, state_goal_list[:, 0, :], save=True, save_path='results/heatmap.mp4')

if __name__ == "__main__":
    main()

