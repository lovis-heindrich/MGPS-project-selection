from typing import NamedTuple
import numpy.typing as npt
import numpy as np
from src.utils.distributions import Distribution


State = tuple[int | float | Distribution, ...]
Tree = list[list[int]]

class Action(NamedTuple):
    expert: int
    query: int


class MouselabConfig(NamedTuple):
    ground_truth: npt.NDArray[np.float64] | None = None
    sample_term_reward: bool = False
    term_belief: bool = True
    limit_repeat_clicks: int | None = 1
    max_actions: int | None = 200

class EpisodeResult(NamedTuple):
    reward: float
    actions: tuple[Action]
    seed: int