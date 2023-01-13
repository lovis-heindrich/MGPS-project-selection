from src.utils.distributions import Normal

def create_tree(num_projects: int, num_criteria: int) -> list[list[int]]:
    """ Creates a tree object used to initialize mouselab_jas environments.

    Args:
        num_projects (int): Number of available projects.
        num_criteria (int): Number of criterias per project.

    Returns:
        list[list[int]]: Tree structure where links from node i are given as a list of adjacent nodes.
    """
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
    """ Initializes Normal distributions from a list of parameters.

    Args:
        mus (list[float]): Mean value of the distribution.
        sigmas (list[float]): Standard deviation of the distribution.

    Returns:
        list[Normal]: List of distributions.
    """
    return [Normal(mu, sigma) for mu, sigma in zip(mus, sigmas)]

