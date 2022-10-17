from src.utils.distributions import Normal
from src.utils.data_classes import MouselabConfig
from src.utils.mouselab_standalone import MouselabJas
import json

def create_tree(num_projects: int, num_criteria: int) -> list[list[int]]:
    assert (num_projects >= 1) and (num_criteria >= 1)
    root = [1 + i*num_criteria for i in range(num_projects)]
    tree = [root]
    for project in range(num_projects):
        subtree = []
        for criteria in range(1, num_criteria + 1):
            if criteria < num_criteria:
                subtree.append([project*num_criteria + criteria + 1])
            else:
                subtree.append([])
        tree.extend(subtree)
    return tree
        
def create_init(mus: list[float], sigmas: list[float]) -> list[Normal]:
    return [Normal(mu, sigma) for mu, sigma in zip(mus, sigmas)]


def create_json(path: str, config: MouselabConfig, num_projects, num_criteria, init: list[Normal], expert_costs: list[float], expert_taus: list[float], seeds: list[int]) -> None:
    structure = {
        "init": [0] + [[state.mu, state.sigma] for state in init[1:]],
        "expert_costs": expert_costs,
        "expert_taus": expert_taus,
        "num_experts": len(expert_costs),
        "num_projects": num_projects,
        "num_criteria": num_criteria
    }
    envs = []
    for seed in seeds:
        env = MouselabJas(num_projects, num_criteria, init, expert_costs, expert_taus, config)
        envs.append({
            "seed": seed,
            "ground_truth": env.ground_truth.tolist(),
            "expert_truth": env.expert_truths.tolist(),
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
