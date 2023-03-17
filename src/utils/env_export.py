""" 
Tools for exporting environments for use in online experiments.
"""

from src.utils.mouselab_jas import MouselabJas
from src.utils.data_classes import MouselabConfig
from src.utils.distributions import Normal

from typing import cast, Any
import json

def create_json(path: str, config: MouselabConfig, seeds: list[int]) -> None:
    """ Creates a json file containing the environment structure and a number of pregenerated environment instances.

    Args:
        path (str): Location where the json file will be stored
        config (MouselabConfig): Mouslab configuration
        seeds (list[int]): List of seeds for which to generate environment instances
    """
    structure = {
        "init": [0] + cast(list[Any], [[state.mu, state.sigma] for state in config.init[1:] if type(state)==Normal]),
        "expert_costs": config.expert_costs,
        "expert_taus": config.expert_taus,
        "num_experts": len(config.expert_costs),
        "num_projects": config.num_projects,
        "num_criteria": config.num_criterias,
        "criteria_scale": config.criteria_scale,
        "max_actions": config.max_actions
    }
    # Check init has been successfully converted
    assert all([type(state)==Normal for state in config.init[1:]])
    assert type(structure["init"])==list
    assert len(structure["init"])==len(config.init)
    envs = []
    for seed in seeds:
        env = MouselabJas(config, seed=seed)
        envs.append({
            "seed": seed,
            "ground_truth": env.ground_truth.tolist(),
            "payoff_matrix": format_payoff(config.num_projects, config.num_criterias, env.expert_truths.tolist())
        })
    data = {"structure": structure, "envs": envs}
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def format_payoff(num_projects: int, num_criteria: int, expert_truths: list[list[float]]) -> list[list[float]]:
    """ Reshapes expert truths from being sorted by expert to being sorted by criteria.

    Args:
        num_projects (int): Number of projects
        num_criteria (int): Number of decision criteria
        expert_truths (list[list[float]]): Array of expert truths

    Returns:
        list[list[float]]: Array of expert truths
    """
    num_experts = len(expert_truths)
    payoff = []
    for criteria in range(num_criteria):
        criteria_index = 1 + criteria
        row = []
        for project in range(num_projects):
            project_index = project * num_criteria
            for expert in range(num_experts):
                row.append(expert_truths[expert][criteria_index + project_index])
        payoff.append(row)
    return payoff