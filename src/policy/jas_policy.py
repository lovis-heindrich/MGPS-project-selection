""" 
Contains the policy base class and some basic baseline policies.
"""

from src.utils.data_classes import Action
from src.utils.mouselab_JAS import MouselabJas
import numpy as np
from abc import ABC, abstractmethod


class JAS_policy(ABC):
    """Policy base class."""

    def __init__(self) -> None:
        pass

    @abstractmethod
    def act(self, env: MouselabJas) -> Action:
        """Return the next action given an environment (including the current environment state).

        Args:
            env (MouselabJas): Environment

        Returns:
            Action: Action taken by the policy.
        """
        pass

    def reset(self) -> None:
        pass


class RandomPolicy(JAS_policy):
    """Policy that takes random actions."""

    def __init__(self) -> None:
        super().__init__()
        self.rng = np.random.default_rng(12345)

    def act(self, env: MouselabJas) -> Action:
        """Act randomly, including randomly terminating.

        Args:
            env (MouselabJas): Environment

        Returns:
            Action: Action taken by the policy.
        """
        actions = tuple(env.actions())
        return actions[self.rng.choice(len(actions))]


class ExhaustivePolicy(JAS_policy):
    """Policy that performs all possible planning actions in order."""

    def act(self, env: MouselabJas) -> Action:
        """Return the next available planning action. Terminates when all possible planning actions have been taken.

        Args:
            env (MouselabJas): Environment

        Returns:
            Action: Action taken by the policy.
        """
        return next(env.actions())


class RandomNPolicy(JAS_policy):
    """Act randomly for a fixed number of actions, then terminate."""

    def __init__(self, N: int) -> None:
        """Stores number of actions to be taken.

        Args:
            N (int): number of actions
        """
        self.N = N
        self.reset()

    def reset(self) -> None:
        """Reset action count."""
        self.current_actions = 0

    def act(self, env: MouselabJas) -> Action:
        """Act randomly if less than N actions have been generated, else terminate.

        Args:
            env (MouselabJas): Environment

        Returns:
            Action: Action taken by the policy.
        """
        if self.current_actions < self.N:
            self.current_actions += 1
            actions = tuple(env.actions())
            actions = tuple(
                [action for action in actions if action is not env.term_action]
            )
            assert env.term_action not in actions
            return actions[np.random.choice(len(actions))]
        else:
            self.reset()
            return env.term_action
