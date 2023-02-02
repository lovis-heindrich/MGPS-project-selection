import numpy as np
from scipy.stats import norm
from src.policy.jas_policy import JAS_policy
from src.utils.data_classes import Action, State
from src.utils.mouselab_jas import MouselabJas
from src.utils.distributions import Normal, expectation

class JAS_voc_policy(JAS_policy):
    def __init__(self, discrete_observations=True) -> None:
        super().__init__()
        self.discrete_observations = discrete_observations

    def act(self, env: MouselabJas) -> Action:
        """ Determines the next action based on the myopic VOC calculation.

        Args:
            env (MouselabJas): Environment

        Returns:
            Action: Action with the highest myopic VOC
        """
        actions = tuple(env.actions())
        values = [self.myopic_voc_normal(env, action) for action in actions]
        costs = [env.cost(action) for action in actions]
        voc = [value + cost for value, cost in zip(values, costs)]
        # Choose randomly between actions with the same voc
        best_action_indices = np.argwhere(voc == np.max(voc)).flatten()
        chosen_action_index = np.random.choice(best_action_indices)
        return actions[chosen_action_index]

    def get_best_path_action_set(self, env: MouselabJas, action: Action, state: State) -> tuple[float, float]:
        """ Expected reward of the best path including and excluding the given action.

        Args:
            action (int): Selected action
            state (list): Environment state

        Returns:
            action_path_reward (float): The expected reward of the best path going through the given action
            alternative_path_reward (float): The expected reward of the best path not going through the given action
        """

        action_path_reward = -np.inf
        alternative_path_reward = -np.inf
        for path in env.all_paths():
            path_reward = 0.
            action_found = False
            for node in path:
                if node == action.query:
                    action_found = True
                path_reward += (expectation(state[node])*env.criteria_scale[node])
            if action_found:
                action_path_reward = max(action_path_reward, path_reward)
            else:
                alternative_path_reward = max(alternative_path_reward, path_reward)
        return action_path_reward, alternative_path_reward

    def myopic_voc_normal(self, env: MouselabJas, action: Action, state: State| None = None, eps=1e-8) -> float:
        """ Computes the voc of a given action.

        Args:
            env (MouselabJas): Environment
            action (Action): Action for which to evaluate the voc
            state (State | None, optional): State. Defaults to None.
            eps (_type_, optional): Tolerance for comparisons. Defaults to 1e-8.

        Returns:
            float: Voc of the given action.
        """
        if self.discrete_observations:
            return self.myopic_voc_normal_discrete(env, action, state, eps)
        else:
            return self.myopic_voc_normal_continuous(env, action, state, eps)

    def myopic_voc_normal_continuous(self, env: MouselabJas, action: Action, state: State| None = None, eps=1e-8) -> float:
        """ Computes the voc of a given action assuming observations are continuous.

        Args:
            env (MouselabJas): Environment
            action (Action): Action for which to evaluate the voc
            state (State | None, optional): State. Defaults to None.
            eps (_type_, optional): Tolerance for comparisons. Defaults to 1e-8.

        Returns:
            float: Voc of the given action.
        """
        assert action in tuple(env.actions()), f"Action {action} not legal"
        if action == env.term_action:
            return 0
        if state == None:
            state = env.state
        assert state is not None
        assert len(state) > action.query
        node = state[action.query]
        assert isinstance(node, Normal)
        action_path_reward, alternative_path_reward = self.get_best_path_action_set(env, action, state)
        action_path_without_node = action_path_reward - node.expectation()
        
        tau = env.expert_taus[action.expert]
        value, sigma = node.mu, node.sigma
        tau_old = 1 / (sigma ** 2)
        tau_new = tau_old + tau
        sample_sigma = 1 / np.sqrt(tau)

        # Sample threshold that leads to a different optimal path
        threshold = (((alternative_path_reward - action_path_without_node)*tau_new) - (value * tau_old)) / tau
        
        # The path leading through the selected node is optimal
        if action_path_reward > alternative_path_reward:
            # Probability of sampling worse than the threshold value
            p_being_worse = norm.cdf(threshold, value, np.sqrt(sigma**2+sample_sigma**2))
            # Expected sample value given that it is below the threshold
            expected_lower = self.expect_lower(value, np.sqrt(sigma**2+sample_sigma**2), threshold)
            # Update the node distribution with the expected sample
            updated_node = ((value * tau_old) + tau * expected_lower) / tau_new
            # Gain = alternative path minus the new node path weighted by probability
            voc = (alternative_path_reward - action_path_without_node - updated_node) * p_being_worse
            if p_being_worse < eps or np.isnan(expected_lower):
                voc = 0
        # The path leading through the selected node is not optimal
        else:
            # Probability of sampling higher than the threshold
            p_being_better = 1 - norm.cdf(threshold, value, np.sqrt(sigma**2+sample_sigma**2))
            # Expected sample value given that it is above the threshold
            expected_higher = self.expect_higher(value, np.sqrt(sigma**2+sample_sigma**2), threshold)
            # Update the node distribution with the expected sample
            updated_node = ((value * tau_old) + tau * expected_higher) / tau_new
            # Gain = new node path minus the old path weighted by probability
            voc = (action_path_without_node + updated_node - alternative_path_reward) * p_being_better
            if p_being_better < eps or np.isnan(expected_higher):
                voc = 0
        return voc

    def myopic_voc_normal_discrete(self, env: MouselabJas, action: Action, state: State| None = None, eps=1e-8) -> float:
        """ Computes the voc of a given action assuming observations are discrete.

        Args:
            env (MouselabJas): Environment
            action (Action): Action for which to evaluate the voc
            state (State | None, optional): State. Defaults to None.
            eps (_type_, optional): Tolerance for comparisons. Defaults to 1e-8.

        Returns:
            float: Voc of the given action.
        """
        assert action in tuple(env.actions()), f"Action {action} not legal"
        if action == env.term_action:
            return 0
        if state == None:
            state = env.state
        assert state is not None
        assert len(state) > action.query
        node = state[action.query]
        node_scale = env.criteria_scale[action.query]
        assert isinstance(node, Normal)
        action_path_reward, alternative_path_reward = self.get_best_path_action_set(env, action, state)
        action_path_without_node = action_path_reward - (node.expectation()*node_scale)
        
        
        tau = env.expert_taus[action.expert]/(node_scale**2)
        value, sigma = node.mu*node_scale, node.sigma*node_scale
        tau_old = 1 / (sigma ** 2)
        tau_new = tau_old + tau
        sample_sigma = 1 / np.sqrt(tau)

        # Sample threshold that leads to a different optimal path
        threshold = ((((alternative_path_reward - action_path_without_node)*tau_new) - (value * tau_old)) / tau) / node_scale
        assert type(env.config.discretize_observations) is tuple
        min_obs, max_obs = env.config.discretize_observations

        # The path leading through the selected node is optimal
        if action_path_reward > alternative_path_reward:
            if threshold < min_obs:
                return 0
            voc = 0
            for obs in range(min_obs, max_obs+1):
                if obs < threshold:
                    p = norm.cdf((obs+0.5)*node_scale, value, np.sqrt(sigma**2+sample_sigma**2)) - norm.cdf((obs-0.5)*node_scale, value, np.sqrt(sigma**2+sample_sigma**2))
                    updated_node = ((value * tau_old) + tau * obs*node_scale) / tau_new
                    if p > eps:
                        voc += (alternative_path_reward - action_path_without_node - updated_node) * p
        # The path leading through the selected node is not optimal
        else:
            if threshold > max_obs:
                return 0
            voc = 0
            for obs in range(min_obs, max_obs+1):
                if obs > threshold:
                    p = norm.cdf((obs+0.5)*node_scale, value, np.sqrt(sigma**2+sample_sigma**2)) - norm.cdf((obs-0.5)*node_scale, value, np.sqrt(sigma**2+sample_sigma**2))
                    updated_node = ((value * tau_old) + tau * obs * node_scale) / tau_new
                    if p > eps:
                        voc += (action_path_without_node + updated_node - alternative_path_reward) * p
        return voc

    def expect_lower(self, mean, sigma, T):
        t = (T-mean)/sigma
        if np.isclose(norm.cdf(t,0,1), 0):
            return np.nan
        return mean - sigma*norm.pdf(t,0,1)/norm.cdf(t,0,1)

    def expect_higher(self, mean, sigma, T):
        t = (T-mean)/sigma
        if np.isclose(norm.cdf(t,0,1), 1):
            return np.nan
        return mean + sigma*norm.pdf(t,0,1)/(1-norm.cdf(t,0,1))