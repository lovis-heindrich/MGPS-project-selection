from src.policy.jas_policy import JAS_policy
from src.utils.mouselab_jas import MouselabJas
from src.utils.data_classes import Action, EpisodeResult
import pandas as pd
import time

def run_episode(env: MouselabJas, policy: JAS_policy, seed: int | None = None) -> EpisodeResult:
    """ Run a single episode using the supplied policy and environment.

    Args:
        env (MouselabJas): Environment
        policy (JAS_policy): Policy
        seed (int | None, optional): Environment seed. Defaults to None.

    Returns:
        EpisodeResult: Results from the simulation.
    """
    env.reset(seed)
    episode_reward = 0
    episode_actions: list[Action] = []
    start_time = time.process_time()
    while not env.done:
        action = policy.act(env)
        _, reward, _, _ = env.step(action)
        episode_reward += reward
        episode_actions.append(action)
    runtime = time.process_time() - start_time
    return EpisodeResult(episode_reward, len(episode_actions), seed, runtime)

def run_simulation(env: MouselabJas, policy: JAS_policy, n=1000, start_seed=None) -> pd.DataFrame:
    """ Run simulations for a number of episodes and aggregate the results.

    Args:
        env (MouselabJas): _description_
        policy (JAS_policy): _description_
        n (int, optional): Number of environment instances. Defaults to 1000.
        start_seed (_type_, optional): If supplied, the seeds from start_seed to start_seed+n will be used. Defaults to None.

    Returns:
        pd.DataFrame: Aggregated results from simulated episodes.
    """
    results: list[EpisodeResult] = []
    if start_seed is not None:
        start, end = start_seed, start_seed + n
    else:
        start, end = 0, n
    for i in range(start, end):
        policy.reset()
        if start_seed is not None:
            result = run_episode(env, policy, i)
        else:
            result = run_episode(env, policy)
        results.append(result)
    return pd.DataFrame(results, columns=EpisodeResult._fields)