from src.utils.data_classes import Action
from src.utils.mouselab_standalone import MouselabJas
import numpy as np

class JAS_policy():
    def __init__(self) -> None:
        pass

    def act(env: MouselabJas) -> Action:
        pass 

class RandomPolicy(JAS_policy):
    def act(env: MouselabJas) -> Action:
        actions = tuple(env.actions())
        return np.random.choice(actions)

class ExhaustivePolicy(JAS_policy):
    def act(env: MouselabJas) -> Action:
        return next(env.actions())