import os
import pickle
import casadi
import torch
import numpy as np
import hypers
from utils import action2waypoints
from env import GridWorld
from networks.gcn import GraphConvNet, GCNPos
from mpc_cbf.robot_unicycle import MPC_CBF_Unicycle
from mpc_cbf.plan_dubins import plan_dubins_path
from utils import dm_to_array
from torch.utils.tensorboard import SummaryWriter
import ipdb
from multi_visual import simulate


import tqdm
from tqdm import trange
def compute_gradient_norm(model, norm_type=2):
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm
def train(world, optim):
    '''
    Define one training episode.
    For each agent, generate one waypoint, following by several MPC steps.
    '''# log_prob:  8 agents 
    # cost:  10 inner steps
    n_explore_steps = 0 # 100 goal/waypoints *
 
    n_inner = 10
    log_probs = torch.zeros([n_explore_steps, n_agents]).to(dev)
    cost_world = torch.zeros([n_explore_steps, n_inner]).to(dev)

    for n in trange(n_explore_steps):
        agents = world.agents
        observations = world.check()
        dists = world.agents_dist()
        ref_states = []
        ub = [size_world[1] * len_grid - 0.1, size_world[0] * len_grid - 0.1, casadi.inf]
        lb = [0, 0, 0]

        # Each agent gets local observations
        for i in range(n_agents):
            agents[i].update_neighbors(dists[i, :])
            agents[i].embed_local(torch.tensor(observations[i], dtype=torch.float32))
        ###########################
        # next single waypoint
        ###########################
        for i in range(n_agents):
            # Get new waypoints
            neighbor_embed = [agents[j].local_embed for j in agents[i].neighbors]    
            actions, log_prob = agents[i].generate_waypoints(observations[i], torch.cat(neighbor_embed, dim=0))
            # print(actions)
            # print(log_prob)
            log_probs[n, i] = log_prob # 计算loss
            xy_goals = action2waypoints(actions, size_world, len_grid)
            # print(xy_goals)
            # print(xy_goals)
            theta_goals = np.random.rand(1) * np.pi - np.pi # Randomly generating theta goal for now.
            state_goal = np.concatenate((xy_goals, theta_goals), axis=-1)

            # Generating ref trajectory dubin路径规划
            path_x, path_y, path_yaw, _, _ = plan_dubins_path(agents[i].states[0], agents[i].states[1], agents[i].states[2],
                                                            state_goal[0], state_goal[1], state_goal[2], r, step_size=v*dt)
            ref_states.append(np.array([path_x, path_y, path_yaw]).T)  # 参考轨迹

        #####################
        # MPC
        #####################
        
        reward_world_list = []
        # cost_agent = torch.tensor([])
        n_inner = np.min([n_inner, len(ref_states[i])])
        for k in range(n_inner): # Multiply MPC steps for each training step
            for i in range(n_agents):
                # Generating MPC trajectory
                u0 = casadi.DM.zeros((agents[i].n_controls, N))
                X0 = casadi.repmat(agents[i].states, 1, N + 1)
                t0 = 0
                # cost_agent = []
                u, X_pred = agents[i].solve(X0, u0, ref_states[i], k, ub, lb)
                t0, X0, u0 = agents[i].shift_timestep(dt, t0, X_pred, u)
                agents[i].states = X0[:, 0]
                # print(agents[i].states)
                if agents[i].states[0] <= size_world[0] and agents[i].states[1] <= size_world[1]:
                    pass
                else:
                    assert agents[i].states[0] <= size_world[0] and agents[i].states[1] <= size_world[1]

                # Log for visualization
                cat_states_list[i] = np.dstack((cat_states_list[i], dm_to_array(X_pred)))

                # cost_agent.append(torch.tensor(world.get_agent_cost(agents[i].id)))
                # cost_agent.append(torch.tensor(agents[i].get_cost_mean()))

            world.step()   #更新世界

            final_cost = torch.tensor(world.get_cost_mean())
            cost_world[n, k] = torch.tensor(world.get_cost_mean())  # -np.mean(self.heatmap * self.cov_lvl)
            # cost_world_list.append(torch.tensor(world.get_cost_max()))
            reward_world_list.append(-world.get_cost_mean())  # didn't use
            heatmaps.append(np.copy(world.heatmap))
            cov_lvls.append(np.copy(world.cov_lvl))
    ############################
    # compute reward
    ############################
    # cost_world: expolore setps 100 * inner steps 10    log_probs: explore steps 100 * agents 8
    reward_delay = torch.tensor(0.0, requires_grad=True).to(dev)
    gamma = torch.tensor(0.8, requires_grad=True).to(dev)
    for i in range(n_explore_steps):
        for j in range(n_inner):
            # 把每一个inner steps的8个agents都乘r倍的cost。
            agents_cost_per_inner_steps = 0
            for q in range(n_agents):
                a = torch.tensor(.0, requires_grad=True).to(dev)
                # 将当前步之后的reward都进行gamma次幂的累乘。离当前步越近的，幂越大，影响越小
                for k in range(n_explore_steps - i, n_explore_steps):
                    a += log_probs[i, q] * (gamma^(k)) * cost_world[i, j]  # 累乘gamma
                agents_cost_per_inner_steps += a
            reward_delay += agents_cost_per_inner_steps
    
    #####################
    # back propogation
    ####################
    # cost_agents = torch.zeros((n_agents,), device=dev) # No costs of individual agent now
    # # loss = torch.matmul(torch.stack(log_probs), cost)
  
    # cost_world = torch.mean(torch.tensor(cost_world_list).to(dev)) 
    # cost = cost_agents + cost_world
    # loss = torch.stack(log_probs).mean() * cost.mean()
    loss = reward_delay
    optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(decisionNN.parameters(), max_norm=1)
    optim.step()
    grad_norm = compute_gradient_norm(decisionNN)

    print("Cost world:%.3f"%reward_delay.detach().cpu().numpy())
    print('loss: %.3f'%loss.detach().cpu().numpy())
    print('Gradient norm: %.3f'%grad_norm)
    print()
    
    writer.add_scalar("Cost/world", reward_delay.detach().cpu().numpy(), epoch)
    # writer.add_scalar("Cost/agent", np.mean(cost_agent_list), epoch)
    writer.add_scalar("Loss/train", loss.detach().cpu().numpy(), epoch)
    writer.add_scalar('Norm/grad', grad_norm)
    writer.flush()
    return reward_delay, -reward_delay


if __name__=='__main__':
    
    with ipdb.launch_ipdb_on_exception():
        #------------------
        # Set training to be deterministic
        #------------------
        seed = 1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        n_agents = 10
        epochs =5
        hypers.init([5, 5, 0.1])
        size_world = (50, 50)
        len_grid = 1
        heatmap = np.ones(size_world) * 0.1
        # heatmap[20:40, 25:45] = 0.6  * np.random.uniform(0, 1, (20, 20))

        world = GridWorld(size_world, len_grid, heatmap, obstacles=None)
        affix = f'agent{n_agents}_epoch{epochs}_corner'
        writer = SummaryWriter(log_dir='runs/overfitting/' + affix)
        
        #------------------
        # Define hyperparameter
        #------------------
        Q_x = 10
        Q_y = 10
        Q_theta = 1
        R_v = 0.5
        R_omega = 0.01
        r_s = 3
        r_c = 10

        dt = 0.1
        N = 20

        r = 3 
        v = 1

        v_lim = [-2, 2]
        omega_lim = [-casadi.pi/4, casadi.pi/4]
        Q = [Q_x, Q_y, Q_theta]
        R = [R_v, R_omega]
        obstacles = [(14,4,3), (18,15,3), (6,19,3), (29, 44,3), (38,15,3), (36,29,3), (25, 26,3)]

        save = True

        # TODO Move the definition of obstacles to env, not in agents. Agents must be able to detect obstacles on the run.
        xy_init = np.random.uniform(0., 50, (n_agents, 2))
        # xy_init[0, :] = np.array([25, 25])
        theta_init= np.random.rand(n_agents, 1)
        state_init = np.concatenate((xy_init, theta_init), axis=-1)
        # t0_list = [0 for i in range(n_agents)]
        agents = [MPC_CBF_Unicycle(i, dt, N, v_lim, omega_lim, Q, R, init_state=state_init[i], obstacles = obstacles, flag_cbf=True, r_s=r_s, r_c=r_c) for i in range(n_agents)]
        # decisionNN = GraphConvNet(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
        decisionNN = GCNPos(hypers.n_embed_channel, size_kernal=3, dim_observe=2*r_s, size_world=size_world, n_rel=hypers.n_rel, n_head=4).to(dev)
        for i in range(n_agents):
            # During the training time, all the agents share the same decision network. Modify this configuration can achieve distributed learning.
            agents[i].decisionNN = decisionNN
        world.add_agents(agents)
        optim = torch.optim.Adam(decisionNN.parameters(), lr=1e-3)
        decisionNN.train()
        # Data for visualization
        cat_states_list = [np.tile(agents[i].states[..., None], (1, N+1)) for i in range(n_agents)]  # 记录状态，nptile复制，（1，N+1）CI
        heatmaps = []
        cov_lvls = []

        #------------------
        # Train
        #------------------
        all_costs = []
        all_rewards = []
        for epoch in range(epochs):
            print(epoch)
            costs, rewards = train(world, optim)
            all_costs.append(costs.detach().cpu().numpy())
            all_rewards.append(rewards)

        net_dict = decisionNN.state_dict()
        log_dict = {'cat_states_list': cat_states_list,
                    'heatmaps': heatmaps,
                    'cov_lvls': cov_lvls,
                    'obstacles': obstacles,
                    'costs': all_costs,  # Save costs
                    'rewards': all_rewards}  # Save rewards}
        
        with open(f'./results/traj_log/train_traj_' + affix + '.pkl', 'wb') as f:
            pickle.dump(log_dict, f)

        if save:
            torch.save({'net_dict': net_dict}, 'results/saved_models/model_' + affix+ '.tar')
        #------------------
        # visual
        #------------------


        heatmap = np.ones(size_world) * 0.1
        heatmap[20:40, 25:45] = 0.6
        world = GridWorld(size_world, len_grid, heatmap, obstacles=None)


        affix = f'agent{n_agents}_epoch{epochs}_corner'
        
        
        with open(f'./results/traj_log/train_traj_' + affix + '.pkl', 'rb') as f:
            log_dict = pickle.load(f)

        # Downsampling the frames
        interval = 1
        cat_states_list = log_dict['cat_states_list']
        for j in range(len(cat_states_list)):
            cat_states_list[j] = cat_states_list[j][:, ::interval] # 10*[3,21]
        heatmaps = log_dict['heatmaps'][::interval]
        cov_lvls =  log_dict['cov_lvls'][::interval]
        obstacles = log_dict['obstacles']
        # Get costs and rewards if available, otherwise use None    
        costs = log_dict.get('costs')
        rewards = log_dict.get('rewards')
        costs = costs[::interval]
        rewards = rewards[::interval]
        n_frames = cat_states_list[0].shape[-1] - 1
        init_states = np.array([cat_states_list[j][:, 0, 0] for j in range(n_agents)])
        simulate(world, n_agents, cat_states_list, heatmaps, cov_lvls, obstacles,
                n_frames, init_states, save=True, cost_values=costs, reward_values=rewards, save_path='results/visual_train.mp4')