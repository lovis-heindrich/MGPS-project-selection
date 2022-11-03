from typing import Generator
import numpy as np
from src.utils.data_classes import MouselabConfig, Action, State
from src.utils.utils import tau_to_sigma, scale_normal
from src.utils.distributions import Distribution, Normal, PointMass, sample, expectation
import warnings

from src.utils.env_creation import create_tree

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
        num_projects: int,
        num_criterias: int,
        init: State,
        expert_costs: list[float],
        expert_taus: list[float],
        config: MouselabConfig,
        criteria_scale: None | list[float] = None,
        seed: None | int = None
    ):
        self.config = config
        self.tree = create_tree(num_projects, num_criterias)
        self.criteria_scale = criteria_scale
        self.num_projects = num_projects

        if self.criteria_scale is None:
            self.init = (0, *init[1:])
        else:
            tmp_scale: list[float] = self.criteria_scale * self.num_projects
            self.init = (0, *[scale_normal(scale, node) for scale, node in zip(tmp_scale, init[1:])])

        # Init costs and precision
        self.num_experts = len(expert_costs)
        self.expert_taus = np.array(expert_taus)
        self.expert_sigma = tau_to_sigma(self.expert_taus)
        self.expert_costs = expert_costs
        self.expert_truths = np.zeros(shape=(self.num_experts, len(self.tree)))

        self.term_action: Action = Action(self.num_experts, len(self.init))

        self.seed = seed
        self.reset()

        assert (
            len(self.ground_truth) == len(self.init) == len(self.state) == len(self.tree)
        ), "state, rewards, and init must be the same length"
        assert len(expert_costs) == len(
            expert_taus
        ), "expert precision and cost arrays must be the same length"

    def reset(self, seed=None) -> State:
        if self.seed is not None:
            np.random.seed(self.seed)
            assert seed is None, "Environment seed already set"
        elif seed is not None:
            np.random.seed(seed)
        self.done = False
        self.clicks: list[Action] = []
        self.state: State = self.init
        if self.config.ground_truth is None:
            self.ground_truth = np.array(list(map(sample, self.init)))
            self.ground_truth[0] = 0.0
        else:
            self.ground_truth = np.array(self.config.ground_truth)
            if self.criteria_scale is not None:
                self.ground_truth[1:] = self.ground_truth[1:] * np.array(self.criteria_scale * self.num_projects)
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
        return self.state

    def cost(self, action: Action) -> float:
        if action is self.term_action:
            return 0
        return -abs(self.expert_costs[action.expert])

    def get_state(self, state: None | State) -> State:
        if state == None:
            state = self.state
        assert state is not None
        return state

    def actions(self, state: None | State = None) -> Generator[Action, None, None]:
        """Yields actions that can be taken in the given state.
        Actions include observing the value of each unobserved node and terminating.
        If a click limit is set only actions fulfilling that condition are returned.
        """
        if self.done:
            return
        state = self.get_state(state)
        if not self.config.max_actions or len(self.clicks) < self.config.max_actions:
            for i, v in enumerate(state):
                for e in range(self.num_experts):
                    if hasattr(v, "sample"):
                        if (
                            self.config.limit_repeat_clicks is None
                            or self.clicks.count(Action(e, i))
                            < self.config.limit_repeat_clicks
                        ):
                            yield Action(e, i)
        yield self.term_action

    def step(self, action: Action) -> tuple[State, float, bool, float]:
        assert not self.done, "terminal state"
        if action == self.term_action:
            self.done = True
            reward = self.term_reward()
            done = True
            obs = 0.0
            return self.state, reward, done, obs
        # Assert that legal action is taken
        assert (self.config.max_actions is None) | (
            len(self.clicks) < self.config.max_actions
        ), "max actions reached"
        assert (self.config.limit_repeat_clicks is None) | (
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
        state = self.get_state(state)
        if self.config.term_belief:
            return self.expected_term_reward(state)  # TODO

        returns = np.array(
            [self.ground_truth[list(path)].sum() for path in self.optimal_paths(state)]
        )
        if self.config.sample_term_reward:
            return float(np.random.choice(returns))
        else:
            return np.mean(returns)

    @lru_cache(CACHE_SIZE)
    def expected_term_reward(self, state: State):
        return self.node_value(0, state).expectation()

    def node_value(self, node: int, state: None | State = None):
        """A distribution over total rewards after the given node."""
        state = self.get_state(state)
        return max(
            (self.node_value(n1, state) + state[n1] for n1 in self.tree[node]),
            default=ZERO,
            key=expectation,
        )

    def optimal_paths(
        self, state: None | State = None, tolerance=0.01
    ) -> Generator[tuple[int, ...], None, None]:
        state = self.get_state(state)

        def rec(path: tuple[int, ...]) -> Generator[tuple[int, ...], None, None]:
            children = self.tree[path[-1]]
            if not children:
                yield path
                return
            quals = [self.node_quality(n1, state).expectation() for n1 in children]
            best_q = max(quals)
            for n1, q in zip(children, quals):
                if np.abs(q - best_q) < tolerance:
                    yield from rec(path + (n1,))

        yield from rec((0,))

    def node_quality(self, node: int, state: None | State = None) -> Distribution:
        """A distribution of total expected rewards if this node is visited."""
        state = self.get_state(state)
        return self.node_value_to(node, state) + self.node_value(node, state)

    def node_value_to(self, node: int, state: None | State = None) -> Distribution:
        """A distribution over rewards up to and including the given node."""
        state = self.get_state(state)
        all_paths = self.path_to(node)
        values: list[float] = []
        path_rewards: list[Distribution] = []
        for path in all_paths:
            path_reward = ZERO
            for n in path:
                if hasattr(n, "sample"):
                    path_reward += state[n]
                else:
                    path_reward += state[n]
            path_rewards.append(path_reward)
            values.append(path_reward.expectation())
        best_path = np.argmax(values)
        return path_rewards[best_path]

    def path_to(self, node: int) -> list[list[int]]:
        """Returns all paths leading to a given node.

        Args:
            node (int): Target node paths to are searched for

        Returns:
            list of list of int: All paths to the node in a nested list
        """
        all_paths = self.all_paths()
        node_paths = [p for p in all_paths if node in p]
        # Cut of remaining path after target node
        up_to_node = [p[: p.index(node) + 1] for p in node_paths]
        return up_to_node

    @lru_cache(CACHE_SIZE)
    def all_paths(self, start=0) -> list[list[int]]:
        def rec(path):
            children = self.tree[path[-1]]
            if children:
                for child in children:
                    yield from rec(path + [child])
            else:
                yield path

        return list(rec([start]))

    def _render(self, mode='notebook', close=False):
        if close:
            return
        from graphviz import Digraph
        
        
        def color(val):
            if val > 0:
                return '#8EBF87'
            else:
                return '#F7BDC4'
        
        dot = Digraph()
        for x, ys in enumerate(self.tree):
            r = self._state[x]
            observed = not hasattr(self._state[x], 'sample')
            c = color(r) if observed else 'grey'
            l = str(round(r, 2)) if observed else str(x)
            dot.node(str(x), label=l, style='filled', color=c)
            for y in ys:
                dot.edge(str(x), str(y))
        return dot
