import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


class Scenario(BaseScenario):
    """
        Diverse behaviour scenario with
            Labeling:
                1) Discrete(3)
                2) Labeled according to its spawn place as left\right\middle
            Custom Policies:
                2) Left agent -> left land.; right agent -> right land.; etc.
    """
    def make_world(self):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        world.labeling = True
        world.custom_policy = False # - seems for action callback, so False == hasattr(self, "custom_policies")
        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.action_callback = None # self.custom_policy
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def labeling(self, agents, mode='spawn'):
        """
        Input:
            agent - object, agent in the world (from make_world)
                Public: agent.state.p_pos, agent.name,
        Output:
            label - int, agent's label - leftmost == 0, rightmost == 2, middle == 1
        """
        if mode == 'random':
            labels = np.random.choice(3, size=3, replace=False)
            for i, agent in enumerate(agents):
                agent.label = labels[i]
        elif mode == "spawn":
            leftmost_i = 0
            rightmost_i = 0
            for i, agent in enumerate(agents):
                if agent.state.p_pos is None:
                    agent.label = 1
                    continue
                agent.label = 1
                if agent.state.p_pos[0] < agents[leftmost_i].state.p_pos[0]:
                    leftmost_i = i
                if agent.state.p_pos[0] > agents[rightmost_i].state.p_pos[0]:
                    rightmost_i = i
            agents[leftmost_i].label = 0
            agents[rightmost_i].label = 2


    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)
        self.labeling(world.agents)


    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)


    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
            #if min(dists) < (world.agents[dists.index(min(dists))].size + l.size) * 1.5:
            #    rew += 10.
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)
        # communication of all other agents
        comm = []
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            comm.append(other.state.c)
            other_pos.append(other.state.p_pos - agent.state.p_pos)
        if world.labeling:
            # {0, 1} - vel, {2, 3} - pos, [{4, 5}, {6, 7}, {8, 9}] - ent_pos, [{10, 11}, {12, 13}] - other_pos, [{14, 15},{16, 17}]
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm +
                                  [np.array(agent.label).reshape(-1)])
        else:
            # {0, 1} - vel, {2, 3} - pos, [{4, 5}, {6, 7}, {8, 9}] - ent_pos, [{10, 11}, {12, 13}] - other_pos, [{14, 15},{16, 17}]
            return np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos + comm)

    def custom_policies(self, obs_n, world):
        """
        Custom policies output
        Out: list of numpy arrays of 1 dimension equal to (2*world.dim_p + 1)
        """
        action_n = []
        for i, (agent, obs) in enumerate(zip(world.agents, obs_n)):
            x, y = agent.state.p_pos
            # ent_pos
            ent_pos = obs[4:10].reshape(-1, 2)
            left_i = np.argmin(ent_pos - agent.state.p_pos, axis=0)[0]  # min x
            right_i = np.argmax(ent_pos - agent.state.p_pos, axis=0)[0]  # max x
            if agent.label == 0: # leftmost landmark
                ent_i = left_i
            elif agent.label == 2: # rightmost landmark
                ent_i = right_i
            else:
                ent_i = (set((0, 1, 2)) - set((left_i, right_i))).pop()
            vec_direct = ent_pos[ent_i]

            # construct action
            action = np.zeros(world.dim_p * 2 + 1)
            # [idle, right, left, up, down]
            if abs(vec_direct[0]) > abs(vec_direct[1]): # for x: go left or go right
                if vec_direct[0] > 0: # go right
                    action[1] = 1
                else: # go left
                    action[2] = 1
            else:
                if vec_direct[1] > 0: # go up
                    action[3] = 1
                else:  # go down
                    action[4] = 1

            #action[np.random.randint(action.shape[0])] = 1
            action_n += [action]
        return action_n

    ### TODO make agent.action_callback(agent, self) in all world.agents # multiagent/core.py
    def custom_policy(self, agent, world):
        """
        Custom policy output
        Out: list of numpy arrays of 1 dimension equal to (2*world.dim_p + 1)
        """
        x, y = agent.state.p_pos
        # ent_pos
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            entity_pos.append(entity.state.p_pos - agent.state.p_pos)
        ent_pos = np.array(entity_pos).reshape(-1, 2)
        left_i = np.argmin(ent_pos - agent.state.p_pos, axis=0)[0]  # min x
        right_i = np.argmax(ent_pos - agent.state.p_pos, axis=0)[0]  # max x
        if agent.label == 0: # leftmost landmark
            ent_i = left_i
        elif agent.label == 2: # rightmost landmark
            ent_i = right_i
        else:
            ent_i = (set((0, 1, 2)) - set((left_i, right_i))).pop()
        vec_direct = ent_pos[ent_i]

        # construct action
        action = np.zeros(world.dim_p * 2 + 1)
        # [idle, right, left, up, down]
        if abs(vec_direct[0]) > abs(vec_direct[1]): # for x: go left or go right
            if vec_direct[0] > 0: # go right
                action[1] = 1
            else: # go left
                action[2] = 1
        else:
            if vec_direct[1] > 0: # go up
                action[3] = 1
            else:  # go down
                action[4] = 1

        return action

        # set env action for a particular agent
