"""不停摆头的原因"""
        theta_goals = np.random.rand(1) * np.pi - np.pi # Randomly generating theta goal for now.
        state_goal = np.concatenate((xy_goals, theta_goals), axis=-1)

        # Generating ref trajectory dubin路径规划
        path_x, path_y, path_yaw, _, _ = plan_dubins_path(agents[i].states[0], agents[i].states[1], agents[i].states[2],
                                                        state_goal[0], state_goal[1], state_goal[2], r, step_size=v*dt)
        ref_states.append(np.array([path_x, path_y, path_yaw]).T)  # 参考轨迹

theta_goals is 随机变量

"""最终聚拢到一起"""
    def generate_waypoints(self, observe, neighbors_observe):
        '''
        Generate waypoints given local observation and neighbors' embeded information
        observe: (1, dim_map, dim_map)
        neighbors_observe: (n_neighbors, dim_embed)
        '''
        observe, neighbors_observe = torch.tensor(observe, dtype=torch.float32, device=self.dev), neighbors_observe.to(self.dev)
        prob = self.decisionNN(observe, neighbors_observe, torch.tensor(self.states, dtype=torch.float32, device=self.dev).view(1, -1))
        # actions = torch.multinomial(prob, 1).cpu().numpy() # Sample one destination for current time step
        # log_prob = torch.log(prob)
        dist = Categorical(prob)  # 分类分布
        action = dist.sample()   # 对分布进行采样
        log_prob = dist.log_prob(action)  # 采取相应动作
        # print(log_prob)
        return action.cpu().numpy(), log_prob
xy_goal只有当前时刻目的地，应该有多个waypoints，或者将waypoints放在在inner循环中。
