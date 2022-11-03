from src.utils.mouselab_jas import MouselabJas
from src.utils.data_classes import MouselabConfig
from src.utils.distributions import Normal

from typing import cast, Any
import json

def create_json(path: str, config: MouselabConfig, num_projects, num_criteria, init: list[Normal], expert_costs: list[float], expert_taus: list[float], criteria_scale: list[float] | None, seeds: list[int]) -> None:
    structure = {
        "init": [0] + cast(list[Any], [[state.mu, state.sigma] for state in init[1:]]),
        "expert_costs": expert_costs,
        "expert_taus": expert_taus,
        "num_experts": len(expert_costs),
        "num_projects": num_projects,
        "num_criteria": num_criteria,
        "criteria_scake": criteria_scale
    }
    envs = []
    for seed in seeds:
        env = MouselabJas(num_projects, num_criteria, init, expert_costs, expert_taus, config, seed=seed)
        envs.append({
            "seed": seed,
            "ground_truth": env.ground_truth.tolist(),
            "payoff_matrix": format_payoff(num_projects, num_criteria, env.expert_truths.tolist())
        })
    data = {"structure": structure, "envs": envs}
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def format_payoff(num_projects, num_criteria, expert_truths):
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