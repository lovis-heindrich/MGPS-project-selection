from src.policy.jas_policy import JAS_policy
from src.utils.mouselab_jas import MouselabJas
from src.utils.data_classes import Action, EpisodeResult
import pandas as pd
import time
import numpy as np
from tqdm import tqdm

def run_episode(env: MouselabJas, policy: JAS_policy, seed: int | None = None, return_obs=False) -> tuple[EpisodeResult, list[Action]] | tuple[EpisodeResult, list[Action], list[float]]:
    """ Run a single episode using the supplied policy and environment.

    Args:
        env (MouselabJas): Environment
        policy (JAS_policy): Policy
        seed (int | None, optional): Environment seed. Defaults to None.

    Returns:
        EpisodeResult: Results from the simulation.
    """
    env.reset(seed)
    episode_reward = 0.
    episode_actions: list[Action] = []
    episode_obs = []
    start_time = time.process_time()
    cost = 0.
    while not env.done:
        action = policy.act(env)
        #print(action)
        _, reward, _, obs = env.step(action)
        episode_reward += reward
        if action!=env.term_action:
            episode_actions.append(action)
            episode_obs.append(obs)
        cost += env.cost(action)
        #print(action, obs)
    expected_reward = env.expected_term_reward(env.state) + cost
    true_reward = np.mean(np.array([sum([env.ground_truth[node]*env.criteria_scale[node] for node in path]) for path in env.optimal_paths(env.state)])) + cost
    runtime = time.process_time() - start_time
    #print(episode_actions)
    if return_obs:
        return EpisodeResult(reward=episode_reward, actions=len(episode_actions), seed=seed, runtime=runtime, true_reward=true_reward, expected_reward=expected_reward), episode_actions, episode_obs
    return EpisodeResult(reward=episode_reward, actions=len(episode_actions), seed=seed, runtime=runtime, true_reward=true_reward, expected_reward=expected_reward), episode_actions

def run_simulation(env: MouselabJas, policy: JAS_policy, n=1000, start_seed=None, use_tqdm=True) -> pd.DataFrame:
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
    all_actions = []
    for i in (tqdm(range(start, end)) if use_tqdm else range(start, end)):
        policy.reset()
        if start_seed is not None:
            result, actions = run_episode(env, policy, i)
        else:
            result, actions = run_episode(env, policy)
        all_actions.append(actions)
        results.append(result)
    return pd.DataFrame(results, columns=EpisodeResult._fields), all_actions