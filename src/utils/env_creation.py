from src.utils.distributions import Normal


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

