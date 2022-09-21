import numpy as np
from scipy.stats import norm
from src.policy.jas_policy import JAS_policy
from src.utils.data_classes import Action, State
from src.utils.mouselab_standalone import MouselabJas

class JAS_voc_policy(JAS_policy):
    def __init__(self) -> None:
        super().__init__()

    def act(self, env: MouselabJas) -> Action:
        actions = tuple(env.actions())
        voc = [self.myopic_voc_normal(env, action) for action in actions]
        best_action_indices = np.argwhere(voc == np.max(voc)).flatten()
        chosen_action_index = np.random.choice(best_action_indices)
        return actions[chosen_action_index]

    def myopic_voc_normal(self, env: MouselabJas, action: Action, state: State| None = None, log=False, eps=1e-8) -> float:
        assert action in tuple(env.actions()), f"Action {action} not legal"
        if action == env.term_action:
            return 0
        if state == None:
            state = env.state
        assert type(state) == State
        assert hasattr(state[action], "sigma")
        action_path_rewards, alternative_path_reward = self.get_best_path_action_set(action, state)
        action_path_without_node = action_path_reward - state[action].expectation()
        
        value, sigma = state[action].mu, state[action].sigma
        tau_old = 1 / (sigma ** 2)
        tau_new = tau_old + self.tau
        sample_sigma = 1 / np.sqrt(self.tau)

        # Sample threshold that leads to a different optimal path
        threshold = (((alternative_path_reward - action_path_without_node)*tau_new) - (value * tau_old)) / self.tau
        
        # The path leading through the selected node is optimal
        if action_path_reward > alternative_path_reward:
            # Probability of sampling worse than the threshold value
            p_being_worse = norm.cdf(threshold, value, np.sqrt(sigma**2+sample_sigma**2))
            # Expected sample value given that it is below the threshold
            expected_lower = self.expect_lower(value, np.sqrt(sigma**2+sample_sigma**2), threshold)
            # Update the node distribution with the expected sample
            updated_node = ((value * tau_old) + self.tau * expected_lower) / tau_new
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
            updated_node = ((value * tau_old) + self.tau * expected_higher) / tau_new
            # Gain = new node path minus the old path weighted by probability
            voc = (action_path_without_node + updated_node - alternative_path_reward) * p_being_better
            if p_being_better < eps or np.isnan(expected_higher):
                voc = 0

        if log:
            print(f"Path value of the chosen action with N({value}, {sigma}): {action_path_reward}. Best alternative path reward: {alternative_path_reward}.")
            print("Threshold", threshold)
            if action_path_reward > alternative_path_reward:
                print(f"Probability of sample to be lower than {threshold}: {p_being_worse}")
                print(f"Expected value lower than {threshold}: {expected_lower}")
            else:
                print(f"Probability of sample to be higher than {threshold}: {p_being_better}")
                print(f"Expected value higher than {threshold}: {expected_higher}")
            print(f"Node value after observe: {updated_node}")
            print(f"Myopic VOC: {voc}")
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