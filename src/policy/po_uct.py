""" 
This script implements PO-UCT from David Silver's "Monte-Carlo planning in large POMDPs" article. 
The MCTS implementation is loosely based on a tutorial created by Tor Lattimore.
"""

from src.policy.jas_policy import JAS_policy
from src.utils.mouselab_jas import MouselabJas
from src.utils.data_classes import Action
import random
import numpy as np

class POUCT_policy(JAS_policy):
    def __init__(self, steps=10000, rollout_depth=5, exploration_coeff=10000, eps=0.00001) -> None:
        self.steps = steps
        self.rollout_depth = rollout_depth
        self.exploration_coeff = exploration_coeff
        self.eps = eps

    def act(self, env: MouselabJas) -> Action:
        pouct = POUCT(env, self.steps, self.rollout_depth, self.exploration_coeff, self.eps)
        action = pouct.search(False)
        return action

    def reset(self) -> None:
        pass

class Node():
    def __init__(self, env: MouselabJas, state, exploration_coeff, eps=0.00001, simulated_actions:None|list[Action]=None):
        """ MCTS search tree node.

        Args:
            env (MouselabPar): Mouselab environment.
            state (_type_): Current belief state.
            exploration_coeff (int): Hyperparameter. Exploration coefficient for UCB action selection.
            eps (float): Initial visit value for MCTS nodes.
        """
        self.simulated_actions: list[Action] = []
        if simulated_actions is not None:
            assert simulated_actions is not None
            self.simulated_actions = simulated_actions
        self.state = state
        self.available_actions = tuple(env.actions(self.state, self.simulated_actions))
        self.N = 1
        self.NA = np.full(len(self.available_actions), eps)
        self.VA = np.zeros(len(self.available_actions))
        self.children: dict[Action, list[Node]] = {a:[] for a in self.available_actions}
        self.C = exploration_coeff
    
    def update(self, action, reward) -> None:
        """ Updates visits and cumulative reward for an action.

        Args:
            action (int): Executed action
            reward (float): Expected reward
        """
        action_index = self.available_actions.index(action)
        self.N += 1
        self.NA[action_index] += 1
        # Rolling mean calculation
        self.VA[action_index] += ((reward - self.VA[action_index]) / self.NA[action_index])
    
    def choose(self) -> Action:
        """ Chooses next action according to UCB-1

        Returns:
            int: Action
        """
        v = self.VA + self.C*np.sqrt(np.log(self.N)/self.NA)
        action = np.argmax(v)
        return self.available_actions[action]

class POUCT():
    def __init__(self, env: MouselabJas, steps=10000, rollout_depth=5, exploration_coeff=10000, eps=0.00001):
        """ PO-UCT algorithm.

        Args:
            env (MouselabPar): Environment at the current state.
            steps (int, optional): Hyperparameter. Number of MCTS steps for each action choice. Defaults to 10000.
            rollout_depth (int, optional): Hyperparameter. Depth of rollout evaluation. Defaults to 5.
            exploration_coeff (int, optional): Hyperparameter. Exploration coefficient for UCB action selection. Defaults to 10000.
            discretize (bool, optional): If true, simulation steps are sampled from a discretized categorical distribution instead of the true Normal distribution. Defaults to True.
            bins (int, optional): Number of bins to use for discretizing Normal distributions. Defaults to 8.
            eps (float, optional): Initial visit value for MCTS nodes. Defaults to 0.00001.
        """
        self.env = env
        self.steps = steps
        self.rollout_depth = rollout_depth
        self.exploration_coeff = exploration_coeff
        self.epsilon = eps

    def search(self, log=False) -> Action:
        """ Performs a MCTS search to find the best action.

        Returns:
            int: Action with the highest expected return.
        """
        node = Node(self.env, self.env.state, self.exploration_coeff, self.epsilon)
        for _ in range(self.steps):
            self.mcts(node)
        action = node.available_actions[np.argmax(node.VA)]
        if log:
            print(f"Action {action} ({np.round(np.max(node.VA), 2)}), term reward {np.round(node.VA[-1], 2)}")
            #print(list(np.round(node.VA, 2)))
            print(list(node.NA.astype(int)))
        #print("Selected", action)
        return action
    
    def rollout(self, obs_state, depth, simulated_actions) -> float:
        """ Perform a random rollout from a given belief state and return the reward. Last action is always the termination action if termination hasn't been selected randomly already.

        Args:
            obs_state (Normal): Current belief state
            depth (int, optional): Number of random actions.

        Returns:
            float: Expected reward after random actions and terminations.
        """
        rewards = 0.
        done = False
        actions = [action for action in simulated_actions]
        while (not done) and (depth>0):
            action = random.choice(list(self.env.actions(obs_state, actions)))
            obs_state, reward, done, _ = self.env.simulate(action, obs_state, actions)
            actions.append(action)
            rewards += reward
            depth -= 1
        if not done:
            _, reward, _, _ = self.env.simulate(self.env.term_action, obs_state, actions)
            rewards += reward
        return rewards

    def mcts(self, node: Node) -> float:
        """ Performs a single MCTS step in a given node.

        Args:
            node (Node): Current MCTS node in which the action is selected.

        Returns:
            float: Expected reward from the MCTS simulation.
        """
        # Choose UCB action
        action = node.choose()
        if self.env.config.limit_repeat_clicks is not None:
            assert type(self.env.config.max_actions) == int
            if len(self.env.clicks) + len(node.simulated_actions) >= self.env.config.max_actions:
                action = self.env.term_action
        if action == self.env.term_action:
            reward = self.env.expected_term_reward(node.state)
            node.update(action, reward)
            return reward
        child_simulated_actions = [prev_action for prev_action in node.simulated_actions] + [action]
        # Sample next state
        obs_state, action_reward, _, _ = self.env.simulate(action, node.state, node.simulated_actions)
        # Check if next state has already been observed
        for child in node.children[action]:
            if (child.state == obs_state) and (set(child.simulated_actions) == set(child_simulated_actions)):
                reward = action_reward + self.mcts(child)
                node.update(action, reward)
                return reward
        # Rollout if state was previously unexplored
        child = Node(self.env, obs_state, self.exploration_coeff, self.epsilon, child_simulated_actions)
        node.children[action].append(child)
        reward = action_reward + self.rollout(obs_state, depth=self.rollout_depth, simulated_actions=child_simulated_actions)
        node.update(action, reward)
        return reward