"""
Custom data classes.
"""

from typing import NamedTuple
from src.utils.distributions import Distribution
from dataclasses import dataclass

State = tuple[int | float | Distribution, ...]
Tree = list[list[int]]

class Action(NamedTuple):
    """ Meta-level actions are specified by the expert from whom information is requested and the tree node. 

    """
    expert: int
    query: int

@dataclass
class MouselabConfig:
    """ Object containing parameters of mouselab_jas environments.
    """
    # Number of project alternatives
    num_projects: int
    # Number of relevant evaluation criteria per project
    num_criterias: int
    # Click cost per expert
    expert_costs: list[float]
    # Precision per expert
    expert_taus: list[float]
    # Initial state distribution
    init: State
    # Importance scaling per criteria
    criteria_scale: None | list[float] = None
    # If true, termination leads to a random path being selected among optimal paths, otherwise the average ground truth is used
    sample_term_reward: bool = False
    # If true, termiation reward is based on the belief state, otherwise the ground truth
    term_belief: bool = True
    # Limit how often experts can be asked for each project and criteria
    limit_repeat_clicks: int | None = 1
    # Overall maximum actions before termination
    max_actions: int | None = 200
    # Discretize expert observations (inclusive range)
    discretize_observations: None | tuple[int, int] = (1, 5)

class EpisodeResult(NamedTuple):
    """ Object containing results of policy simulations.
    """
    reward: float
    actions: int
    seed: int | None
    runtime: float
    true_reward: float
    expected_reward: float