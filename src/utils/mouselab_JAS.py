""" 
Main mouselab environment adapted to the project selection choice task.
"""

from typing import Any, Generator
import numpy as np
from src.utils.data_classes import MouselabConfig, Action, State
from src.utils.utils import tau_to_sigma
from src.utils.env_creation import create_tree
from src.utils.distributions import Normal, PointMass, sample, expectation
import warnings
import numpy.typing as npt

NO_CACHE = False
if NO_CACHE:
    lru_cache = lambda _: (lambda f: f)
else:
    from functools import lru_cache
CACHE_SIZE = int(2**14)

ZERO = PointMass(0)


class MouselabJas:
    def __init__(
        self,
        config: MouselabConfig,
        ground_truth: npt.NDArray[np.float64] | None = None,
        seed: None | int = None,
    ):
        self.config = config
        self.tree = create_tree(config.num_projects, config.num_criterias)
        self.num_projects = config.num_projects

        # Criteria scale is converted to dictionary format
        if config.criteria_scale is None:
            self.criteria_scale = {i: 1.0 for i in range(len(self.tree))}
        else:
            node_scale = [1] + config.criteria_scale * self.num_projects
            assert len(node_scale) == len(self.tree)
            self.criteria_scale = {i: weight for i, weight in enumerate(node_scale)}
        # Initial ground truth value stored for resetting the environment
        self.init_ground_truth = ground_truth

        self.init = (0, *config.init[1:])

        # Init costs and precision
        self.num_experts = len(config.expert_costs)
        self.expert_taus = np.array(config.expert_taus)
        self.expert_sigma = tau_to_sigma(self.expert_taus)
        self.expert_costs = config.expert_costs
        self.expert_truths = np.zeros(shape=(self.num_experts, len(self.tree)))

        self.term_action: Action = Action(self.num_experts, len(self.init))

        self.seed = seed

        # Creates ground truth and expert samples
        self.reset()

        assert (
            len(self.ground_truth)
            == len(self.init)
            == len(self.state)
            == len(self.tree)
        ), "state, rewards, and init must be the same length"
        assert len(config.expert_costs) == len(
            config.expert_taus
        ), "expert precision and cost arrays must be the same length"

    def reset(self, seed: int | None = None) -> State:
        """Resets the environment by resetting the state as well as generating new ground truth rewards and expert samples.

        Args:
            seed (int, optional): Environment seed. Defaults to None.

        Returns:
            State: New initial environment state
        """
        if self.seed is not None:
            np.random.seed(self.seed)
            assert seed is None, "Environment seed already set"
        elif seed is not None:
            np.random.seed(seed)
        self.done = False
        self.clicks: list[Action] = []
        self.state: State = self.init
        if self.init_ground_truth is None:
            self.ground_truth = np.array(list(map(sample, self.init)))
            self.ground_truth[0] = 0.0
        else:
            self.ground_truth = np.array(self.init_ground_truth)
            if self.ground_truth[0] != 0:
                warnings.warn("ground_truth[0] will be set to 0", UserWarning)
            self.ground_truth[0] = 0.0
        # Resample experts
        for expert in range(self.num_experts):
            sigma = self.expert_sigma[expert]
            dist = [Normal(truth, sigma) for truth in self.ground_truth[1:]]
            self.expert_truths[expert, :] = [self.ground_truth[0]] + [
                s.sample() for s in dist
            ]
            if self.config.discretize_observations is not None:
                min_obs, max_obs = self.config.discretize_observations
                self.expert_truths = np.rint(self.expert_truths)
                self.expert_truths[self.expert_truths < min_obs] = min_obs
                self.expert_truths[self.expert_truths > max_obs] = max_obs
        return self.state

    def action_to_readable(self, action: Action) -> tuple[int, int, int]:
        """ Converts an action object to a [project, criteria, expert] tuple.

        Args:
            action (Action): Input Action object

        Returns:
            tuple[int, int, int]: Tuple of [project, criteria, expert]
        """
        expert = action.expert
        criteria = (action.query - 1) % self.config.num_criterias
        project = (action.query - 1) // self.config.num_criterias
        #print(f"Project {project}, Criteria {criteria}, Expert {expert}")
        return project, criteria, expert

    def cost(self, action: Action) -> float:
        """Returns the cost of the given action.

        Args:
            action (Action): Selected action

        Returns:
            float: Cost
        """
        if action is self.term_action:
            return 0
        return -abs(self.expert_costs[action.expert])

    def get_state(self, state: None | State) -> State:
        """Helper function that returns the given state. If no state is given the current environment state is returned.

        Args:
            state (None | State): State

        Returns:
            State: State
        """
        if state == None:
            state = self.state
        assert state is not None
        return state

    def actions(self, state: None | State = None, simulated_actions: list[Action] | None= None) -> Generator[Action, None, None]:
        """Yields actions that can be taken in the given state.
        Actions include observing the value of each unobserved node and terminating.
        If a click limit is set only actions fulfilling that condition are returned.
        """
        if self.done:
            return
        if simulated_actions is None:
            simulated_actions = []
        assert simulated_actions is not None
        click_counts = self.clicks + simulated_actions
        state = self.get_state(state)
        if (not self.config.max_actions) or (len(self.clicks) < self.config.max_actions):
            for i, v in enumerate(state):
                for e in range(self.num_experts):
                    if hasattr(v, "sample"):
                        if (
                            self.config.limit_repeat_clicks is None
                            or click_counts.count(Action(e, i))
                            < self.config.limit_repeat_clicks
                        ):
                            yield Action(e, i)
        yield self.term_action

    def step(self, action: Action) -> tuple[State, float, bool, float]:
        """Performs an action and updates the environment state accordingly.

        Args:
            action (Action): Action to be taken.

        Returns:
            tuple[State, float, bool, float]: New state, reward, done, observation
        """
        assert not self.done, "terminal state"
        if action == self.term_action:
            self.done = True
            reward = self.term_reward()
            done = True
            obs = 0.0
            return self.state, reward, done, obs
        # Assert that legal action is taken
        if self.config.max_actions != None:
            assert type(self.config.max_actions) == int
            assert len(self.clicks) < self.config.max_actions, "max actions reached"
        if self.config.limit_repeat_clicks is not None:
            assert type(self.config.limit_repeat_clicks) == int
            assert (
                self.clicks.count(action) < self.config.limit_repeat_clicks
            ), "max repeat clicks reached"
        assert hasattr(self.state[action.query], "sample"), "state already observed"
        # Observe a new node
        self.clicks.append(action)
        self.state, obs = self.observe(action)
        reward = self.cost(action)
        done = False
        return self.state, reward, done, obs

    def observe(self, action: Action) -> tuple[State, float]:
        """Observes a state in a partially observable environment by updating the Normal distribution with the new observation

        Args:
            action (int): The action performed

        Returns:
            Tuple: Updated state after observation
        """
        node_state = self.state[action.query]
        assert isinstance(
            node_state, Normal
        ), "only Normal distributions are supported as of now"
        tau_old = 1 / (node_state.sigma**2)
        obs = self.expert_truths[action.expert, action.query]
        tau_new = tau_old + self.expert_taus[action.expert]
        mean_new = (
            (node_state.mu * tau_old) + self.expert_taus[action.expert] * obs
        ) / tau_new
        sigma_new = 1 / np.sqrt(tau_new)
        s = list(self.state)
        s[action.query] = Normal(mean_new, sigma_new)
        return tuple(s), obs

    def term_reward(self, state: None | State = None) -> float:
        """Termination reward in the given state.

        Args:
            state (None | State, optional): State. Defaults to None.

        Returns:
            float: Termination reward.
        """
        state = self.get_state(state)
        if self.config.term_belief:
            return self.expected_term_reward(state)
        else:
            return self.true_term_reward(state)

    def true_term_reward(self, state: State) -> float:
        """True termination reward (based on ground truth values) in a given state.

        Args:
            state (State): State

        Returns:
            float: Termination reward
        """
        returns = np.array(
            [
                self.path_value(list(path)) for path in self.optimal_paths(state)
            ]
        )

        if self.config.sample_term_reward:
            return float(np.random.choice(returns))
        else:
            return float(np.mean(returns))

    @lru_cache(CACHE_SIZE)
    def expected_term_reward(self, state: State) -> float:
        """Expected termination reward (based on current belief state) in a given state

        Args:
            state (State): State

        Returns:
            float: Termination reward
        """
        # Return first generator element
        for path in self.optimal_paths(state):
            return self.expected_path_value(list(path), state)
        assert False, "Should never happen"
        return 0.0
    
    def path_value(self, path: list[int]) -> float:
        """Return the summed values of all nodes on a given path.

        Args:
            path (list[int]): List of nodes visited.
            state (State): State

        Returns:
            float: Value of visiting all nodes on the path
        """
        return sum([
                self.ground_truth[node] * self.criteria_scale[node]
                for node in path
            ])
        

    def expected_path_value(self, path: list[int], state: State) -> float:
        """Return the summed expected values of all nodes on a given path.

        Args:
            path (list[int]): List of nodes visited.
            state (State): State

        Returns:
            float: Expected value of visiting all nodes on the path
        """
        return sum(
            [expectation(state[node]) * self.criteria_scale[node] for node in path]
        )

    def optimal_paths(
        self, state: None | State = None, tolerance=0.01
    ) -> Generator[tuple[int, ...], None, None]:
        """Finds all paths through the environment that are optimal according to a given belief state.

        Args:
            state (None | State, optional): State. Defaults to the current state.
            tolerance (float, optional): Tolerance for comparing path rewards. Defaults to 0.01.

        Yields:
            Generator[tuple[int, ...], None, None]: Optimal paths.
        """
        state = self.get_state(state)
        paths = list(self.all_paths())
        path_values = [self.expected_path_value(path, state) for path in paths]
        max_path = np.max(path_values)
        for value, path in zip(path_values, paths):
            if np.abs(max_path - value) < tolerance:
                yield path

    @lru_cache(CACHE_SIZE)
    def all_paths(self, start=0) -> list[list[int]]:
        """Finds all possible paths from a given starting node.

        Args:
            start (int, optional): Node from which paths are generated. Defaults to the root node (0).

        Returns:
            list[list[int]]: List of all paths.
        """

        def rec(path):
            children = self.tree[path[-1]]
            if children:
                for child in children:
                    yield from rec(path + [child])
            else:
                yield path

        return list(rec([start]))

    def _render(self) -> Any:
        """Creates a graph diagram of the environment structure. Import is handled in the function as the library has complicated installation requirements.

        Returns:
            Digraph: Digraph diagram
        """
        from graphviz import Digraph

        def color(val):
            if val > 0:
                return "#8EBF87"
            else:
                return "#F7BDC4"

        dot = Digraph()
        for x, ys in enumerate(self.tree):
            r = self.state[x]
            if type(r) != int:
                r = r.mu
            clicks = [click.query for click in self.clicks]
            observed = clicks.count(x) > 0
            if not observed:
                l = str(x)
            elif type(r) == float:
                l = str(round(r, 2))
            else:
                l = str(r)
            c = color(r) if observed else "grey"
            dot.node(str(x), label=l, style="filled", color=c)
            for y in ys:
                dot.edge(str(x), str(y))
        return dot
