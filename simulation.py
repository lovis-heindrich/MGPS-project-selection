from src.policy.jas_policy import JAS_policy
from src.utils.mouselab_standalone import MouselabJas
from src.utils.data_classes import Action, EpisodeResult
import pandas as pd

def run_episode(env: MouselabJas, policy: JAS_policy, seed: int | None = None):
    env.reset(seed)
    episode_reward = 0
    episode_actions: list[Action] = []
    while not env.done:
        action = policy.act(env)
        _, reward, _, _ = env.step(action)
        episode_reward += reward
        episode_actions.append(action)
    return EpisodeResult(episode_reward, tuple(episode_actions), seed)

def run_simulation(env: MouselabJas, policy: JAS_policy, n=1000, start_seed=None):
    results: list[EpisodeResult] = []
    if start_seed is not None:
        start, end = start_seed, start_seed + n
    else:
        start, end = 0, n
    for i in range(start, end):
        if start_seed is not None:
            result = run_episode(env, policy, i)
        else:
            result = run_episode(env, policy)
        results.append(result)
    return pd.DataFrame(results, columns=EpisodeResult._fields)