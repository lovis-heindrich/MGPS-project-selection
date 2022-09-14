
import random
import math
import numpy as np
from functools import lru_cache
from scipy.stats import norm
from toolz import get, compose, memoize
from src.utils.mouselab_VAR import MouselabVar
from src.utils.distributions import Normal, sample
from src.utils.utils import tau_to_sigma


def hash_key(args, kwargs):
    state = args[1]
    hash_state = lambda state, conf: hash(state) + hash(conf)
    if state is None:
        return state
    elif len(args) == 4: #V
        n = args[2]
        bins = args[3]
        conf = str(n) + "+" + str(bins)
        return hash_state(state, conf)
    # elif len(args) == 5: #Q
    else:
        if kwargs:
            # Blinkered approximation. Hash key is insensitive
            # to states that can't be acted on, except for the
            # best expected value
            # Embed the action subset into the state.
            action_subset = kwargs['action_subset']
            mask = [0] * len(state)
            for a in action_subset:
                mask[a] = 1
            state = tuple(zip(state, mask))
        return hash_state(state)

def expect_lower(mean, sigma, T):
    t = (T-mean)/sigma
    if np.isclose(norm.cdf(t,0,1), 0):
        return np.nan
    return mean - sigma*norm.pdf(t,0,1)/norm.cdf(t,0,1)

def expect_higher(mean, sigma, T):
    t = (T-mean)/sigma
    if np.isclose(norm.cdf(t,0,1), 1):
        return np.nan
    return mean + sigma*norm.pdf(t,0,1)/(1-norm.cdf(t,0,1))

class MouselabJas(MouselabVar):
    """MetaMDP for a tree with a discrete unobserved reward function."""
    metadata = {'render.modes': ['human', 'array']}
    term_state = '__term_state__'

    def __init__(self, tree, init, ground_truth=None, cost=0,
                sample_term_reward=False, term_belief=True, tau=1, repeat_cost=1, myopic_mode="normal", limit_clicked_nodes=None, limit_repeat_clicks=1, samples=None, max_actions=None, expert_costs=None, expert_taus=None):

        self.num_experts = len(expert_costs)
        self.expert_taus = np.array(expert_taus)#np.array([0.1, 0.01, 0.001])
        self.expert_sigma = tau_to_sigma(self.expert_taus)
        self.expert_costs = expert_costs#[1, 0.5, 0.1]
        
        
        super().__init__(tree, init, ground_truth=ground_truth, cost=cost, term_belief=term_belief, sample_term_reward=sample_term_reward, simple_cost=True)

        self.expert_truths = np.zeros(shape=(self.num_experts, len(tree)))
        for expert in range(self.num_experts):
            sigma = self.expert_sigma[expert]
            dist = [Normal(truth, sigma) for truth in self.ground_truth[1:]]
            self.expert_truths[expert, :] = [self.ground_truth[0]] + [s.sample() for s in dist]

        self.cost = lambda action: -abs(self.expert_costs[action[0]])

        self.term_action = 0
        self.max_actions = max_actions
        self.n_actions = self.action_space.n-1
        self.n_obs = len(self.init)-1

        # Precision for partial observations
        self.tau = tau 
        self.repeat_cost= - abs(repeat_cost)

        # Overwrite Myopic VOC based on mode
        if myopic_mode == "normal":
            self.myopic_fun = lambda action, state: self.myopic_voc_normal(action, state)
        elif myopic_mode == "discrete":
            self.myopic_fun = lambda action, state: self.myopic_voc_discrete(action, state)
        else:
            self.myopic_fun = lambda action, state: self.myopic_voc(action, self.discretize(state, bins=4))
        self.myopic_mode = myopic_mode
        self.n_steps = [1, 2, 3, 5, 10]

        # Stores all clicks
        self.clicks = []
        # Optional: Limit the number of different nodes that can be clicked
        # This allows to constrain strategies to human working memory limitations
        self.limit_clicks=limit_clicked_nodes
        # Optional: Limit the number of times each node can be clicked
        # This allows to represent memory constraints of how many samples humans can remember
        self.limit_repeat_clicks=limit_repeat_clicks

        # Fix observations with a dictionary of actions:[samples]
        self.samples = samples

        self._reset()

    def _reset(self, reset_ground_truth=False):
        if self.initial_states:
            self.init = random.choice(self.initial_states)
        self._state = self.init
        obs_list = {}
        for i in range(len(self.init)):
            for e in range(self.num_experts):
                obs_list[(e,i)] = []
        self.obs_list = obs_list
        self.clicks = []
        if reset_ground_truth:
            self.ground_truth = np.array(list(map(sample, self.init)))
            self.ground_truth[0] = 0.
        return self._state

    def actions(self, state):
        """Yields actions that can be taken in the given state.
        Actions include observing the value of each unobserved node and terminating.
        If a click limit is set only actions fulfilling that condition are returned.
        """
        if state is self.term_state:
            return
        if not self.max_actions or len(self.clicks) < self.max_actions:
            for i, v in enumerate(state):
                for e in range(self.num_experts):
                    if hasattr(v, 'sample'):
                        available = True
                        if self.limit_clicks is not None:
                            # If the limit of unique actions has been reached only allow repeat clicks
                            if (len(np.unique(self.clicks)) >= self.limit_clicks) and (i not in self.clicks):
                                available = False
                        if self.limit_repeat_clicks is not None:
                            # Only allow clicks which haven't reached the repeat limit
                            if self.clicks.count((e, i)) >= self.limit_repeat_clicks:
                                available = False
                        if available:
                            yield (e, i)
        yield self.term_action

    def myopic_action_feature(self, action):
        if action == self.term_action:
            return [0, 0]
        else:
            return [self.myopic_fun(action, self._state), self.cost(action)]

    def get_best_path_action_set(self, action, state):
        """ Expected reward of the best path including and excluding the given action.

        Args:
            action (int): Selected action
            state (list): Environment state

        Returns:
            action_path_reward (float): The expected reward of the best path going through the given action
            alternative_path_reward (float): The expected reward of the best path not going through the given action
        """
        assert len(action) == 2, "Adapt for new action format (expert, action)"
        action_path_reward = -np.inf
        alternative_path_reward = -np.inf
        for path in self.all_paths_:
            path_reward = 0
            action_found = False
            for node in path:
                if node == action[1]:
                    action_found = True
                reward = state[node]
                if hasattr(reward, "sample"):
                    path_reward += reward.expectation()
                else:
                    path_reward += reward
            if action_found:
                action_path_reward = max(action_path_reward, path_reward)
            else:
                alternative_path_reward = max(alternative_path_reward, path_reward)
        return action_path_reward, alternative_path_reward

    def myopic_voc_normal(self, action, state, log=False, eps=1e-8):
        """ Calculates the myopic VOC for a given action.

        Args:
            action (int): Selected action
            state (list): Evaluated environment state

        Returns:
            voc (float): Myopic value of computation
        """
        if action == self.term_action:
            return 0
        assert hasattr(state[action], "sigma")
        action_path_reward, alternative_path_reward = self.get_best_path_action_set(action, state)
        action_path_without_node = action_path_reward - state[action].expectation()
        
        value, sigma = state[action].mu, state[action].sigma
        tau_old = 1 / (sigma ** 2)
        tau_new = tau_old + self.tau
        sample_sigma = 1 / math.sqrt(self.tau)

        # Sample threshold that leads to a different optimal path
        threshold = (((alternative_path_reward - action_path_without_node)*tau_new) - (value * tau_old)) / self.tau
        
        # The path leading through the selected node is optimal
        if action_path_reward > alternative_path_reward:
            # Probability of sampling worse than the threshold value
            p_being_worse = norm.cdf(threshold, value, np.sqrt(sigma**2+sample_sigma**2))
            # Expected sample value given that it is below the threshold
            expected_lower = expect_lower(value, np.sqrt(sigma**2+sample_sigma**2), threshold)
            # Update the node distribution with the expected sample
            updated_node = ((value * tau_old) + self.tau * expected_lower) / tau_new
            # Gain = alternative path minus the new node path weighted by probability
            voc = (alternative_path_reward - action_path_without_node - updated_node) * p_being_worse
            if p_being_worse < eps or math.isnan(expected_lower):
                voc = 0
        # The path leading through the selected node is not optimal
        else:
            # Probability of sampling higher than the threshold
            p_being_better = 1 - norm.cdf(threshold, value, np.sqrt(sigma**2+sample_sigma**2))
            # Expected sample value given that it is above the threshold
            expected_higher = expect_higher(value, np.sqrt(sigma**2+sample_sigma**2), threshold)
            # Update the node distribution with the expected sample
            updated_node = ((value * tau_old) + self.tau * expected_higher) / tau_new
            # Gain = new node path minus the old path weighted by probability
            voc = (action_path_without_node + updated_node - alternative_path_reward) * p_being_better
            if p_being_better < eps or math.isnan(expected_higher):
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

    def myopic_voc_discrete(self, action, state, bins=4):
        """ Myopic VOC computation based on discretized normal distribution states.

        Args:
            action (int): Selected action
            state (list): Evaluated environment state

        Returns:
            voc (float): Myopic value of computation
        """
        # Discretized implementation
        state = self.discretize(state, bins=bins)
        node = state[action]
        term_reward = self.expected_term_reward(state)
        voc = 0
        for p, v in zip(node.probs, node.vals):
            # Update distribution of observed node
            obs_state = self._observe_voc(action, self._state, v)
            new_term_reward = self.expected_term_reward(obs_state)
            voc += (new_term_reward - term_reward) * p
        return voc

    def dp_discrete(self, state, action=None, n=1, bins=4):
        """ Approximate DP computation based on discretized normal distribution states.

        Args:
            action (int): Selected action
            state (list): Evaluated environment state

        Returns:
            voc (float): Myopic value of computation
        """

        
        # Discretized implementation
        if action == None:
            return self.v_rec(tuple(state), n, bins)
        elif action == self.term_action:
            return self.expected_term_reward(state)
        else:
            return self.q_rec(tuple(state), action, n, bins)
    
    @memoize(key = hash_key)
    def v_rec(self, state, n, bins):
        actions = self.actions(state)
        if n == 0:
            return self.expected_term_reward(state)
        else: 
            qs = []
            for action in actions:
                if action == self.term_action:
                    qs.append(self.expected_term_reward(state))
                else:
                    q = self.q_rec(state, action, n, bins) 
                    qs.append(q)
            return max(qs)

    def q_rec(self, state, action, n, bins):
        state_disc = self.discretize(state, bins=bins)
        node = state_disc[action]
        q_val = 0
        for p, v in zip(node.probs, node.vals):
            # Update distribution of observed node
            new_state = self._observe_voc(action, state, v)
            q_val += (self.v_rec(new_state, n-1, bins) * p)
        return q_val + self.cost(action)

    def _observe_voc(self, action, state, obs):
        # Update distribution without a given observation instead of random sample
        state = [s for s in state]
        expert, query = action
        mean_old = state[query].mu
        sigma_old = state[query].sigma
        tau_old = 1 / (sigma_old ** 2)
        tau_new = tau_old + self.expert_taus[expert]
        mean_new = ((mean_old * tau_old) + self.expert_taus[expert] * obs) / tau_new
        sigma_new = 1 / math.sqrt(tau_new)
        state[action] = Normal(mean_new, sigma_new)
        return tuple(state)

    def _step(self, action, obs=None):
        if self._state is self.term_state:
            assert 0, 'state is terminal'
        if action == self.term_action:
            reward = self._term_reward()
            done = True
            obs=False
            return self._state, reward, done, obs
        expert, query = action
        if (self.limit_repeat_clicks is not None and self.clicks.count((expert, query)) >= self.limit_repeat_clicks) or not hasattr(self._state[query], 'sample'): # already observed
            self.clicks.append(action)
            reward = self.repeat_cost
            done = False
            obs=True
        else:  # observe a new node
            # Get observation from list
            if self.samples is not None:
                if len(self.samples[action]) > 0:
                    obs = self.samples[action].pop(0)
                else:
                    raise Exception(f"Sample-list {action} is empty")
            self.clicks.append(action)
            self._state = self._observe(action, obs=obs)
            reward = self.cost(action)
            done = False
            obs=self.obs_list[action][-1]
            #print(f"Updated node {action} with mean {self._state[action].mu} sigma {self._state[action].sigma} and sample {obs}")
        return self._state, reward, done, obs

    def simulate(self, action, state, discretize=False, bins=4):
        if state is self.term_state:
            assert 0, 'state is terminal'
        if action == self.term_action:
            reward = self.expected_term_reward(state)
            done = True
            obs = False
            return state, reward, done, obs
        elif not hasattr(state[action], 'sample'):  # already observed
            raise Exception(f"State {action} not a distribution.")
            reward = self.repeat_cost
            done = False
            obs=True
        else:  # observe a new node
            if discretize:
                state_disc = state[action].to_discrete(n=bins, max_sigma=2)
                obs = state_disc.sample()
            else:
                obs = state[action].sample()
            mean_old = state[action].mu
            sigma_old = state[action].sigma
            tau_old = 1 / (sigma_old ** 2)
            tau_new = tau_old + self.tau
            mean_new = ((mean_old * tau_old) + self.tau * obs) / tau_new
            sigma_new = 1 / math.sqrt(tau_new)
            state = tuple([s if i != action else Normal(mean_new, sigma_new) for i, s in enumerate(state)])
            reward = self.cost(action)
            done = False
        return state, reward, done, obs

    # Discretize before calculating features
    def action_features(self, action, bins=4, state=None):
        """Returns the low action features used for BMPS

        Arguments:
            action: low level action for computation
            option: option for which computation
            bins: number of bins for discretization
            state: low state for computation
        """
        state = state if state is not None else self._state
        state_disc = self.discretize(state, bins)

        assert state is not None

        if action == self.term_action:
            if self.simple_cost:
                return np.array([
                    0,
                    0,
                    0,
                    0,
                    self.expected_term_reward(state)
                ])
            else:
                return np.array([
                    [0, 0, 0],
                    0,
                    0,
                    0,
                    self.expected_term_reward(state)
                ])

        if self.simple_cost:
            return np.array([
                self.cost(action),
                self.myopic_fun(action, state),
                self.vpi_action(action, state_disc),
                self.vpi(state_disc),
                self.expected_term_reward(state)
            ])

        else:
            return np.array([
                self.action_cost(action),
                self.myopic_fun(action, state),
                self.vpi_action(action, state_disc),
                self.vpi(state_disc),
                self.expected_term_reward(state)
            ])

    def _observe(self, action, obs=None):
        """ Observes a state in a partially observable environment by updating the Normal distribution with the new observation

        Args:
            action (int): The action performed

        Returns:
            Tuple: Updated state after observation
        """
        expert, query = action
        mean_old = self._state[query].mu
        sigma_old = self._state[query].sigma
        tau_old = 1 / (sigma_old ** 2)
        if obs==None:
            obs = self.expert_truths[expert, query]
        self.obs_list[action].append(obs)
        tau_new = tau_old + self.expert_taus[expert]
        mean_new = ((mean_old * tau_old) + self.expert_taus[expert] * obs) / tau_new
        sigma_new = 1 / math.sqrt(tau_new)
        s = list(self._state)
        s[query] = Normal(mean_new, sigma_new)
        return tuple(s)

    def _observe_voc(self, action, state, obs):
        expert, query = action
        state = [s for s in state]
        mean_old = state[query].mu
        sigma_old = state[query].sigma
        tau_old = 1 / (sigma_old ** 2)
        tau_new = tau_old + self.expert_taus[expert]
        mean_new = ((mean_old * tau_old) + self.expert_taus[expert] * obs) / tau_new
        sigma_new = 1 / math.sqrt(tau_new)
        state[action] = Normal(mean_new, sigma_new)
        return tuple(state)

    @classmethod
    def new_symmetric(cls, branching, reward, repl_init=None, seed=None, **kwargs):
        """Returns a MouselabEnv with a symmetric structure.

        Arguments:
            branching: a list that specifies the branching factor at each depth.
            reward: a function that returns the reward distribution at a given depth."""
        if seed is not None:
            np.random.seed(seed)
        if not callable(reward):
            r = reward
            reward = lambda depth: r

        init = []
        tree = []

        def expand(d):
            my_idx = len(init)
            init.append(reward(d))
            children = []
            tree.append(children)
            for _ in range(get(d, branching, 0)):
                child_idx = expand(d+1)
                children.append(child_idx)
            return my_idx

        expand(0)
        if repl_init is not None: #TODO Check changes
            init = repl_init
        return cls(tree, init, **kwargs)
