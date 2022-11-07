from src.utils.data_classes import Action
from src.utils.mouselab_jas import MouselabJas
import numpy as np

class JAS_policy():
    def __init__(self) -> None:
        pass

    def act(self, env: MouselabJas) -> Action:
        pass 

    def reset(self) -> None:
        pass

class RandomPolicy(JAS_policy):
    def act(self, env: MouselabJas) -> Action:
        actions = tuple(env.actions())
        return actions[np.random.choice(len(actions))]

class ExhaustivePolicy(JAS_policy):
    def act(self, env: MouselabJas) -> Action:
        return next(env.actions())

class RandomNPolicy(JAS_policy):
    def __init__(self, N) -> None:
        self.N = N
        self.reset()
    
    def reset(self) -> None:
        self.current_actions = 0

    def act(self, env: MouselabJas) -> Action:
        if self.current_actions < self.N:
            self.current_actions += 1
            actions = tuple(env.actions())
            actions = tuple([action for action in actions if action is not env.term_action])
            assert env.term_action not in actions
            return actions[np.random.choice(len(actions))]
        else:
            self.reset()
            return env.term_action
