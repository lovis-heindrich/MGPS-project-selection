from src.utils.env_creation import create_init
from src.utils.mouselab_jas import MouselabJas
from src.utils.data_classes import MouselabConfig
from src.utils.utils import sigma_to_tau

def get_env(num_projects=5) -> tuple[MouselabJas, MouselabConfig]:
    # Fixed from paper
    num_criteria = 6
    weights = [0.0206795, 0.0672084, 0.2227102, 0.1067428, 0.4665054, 0.1161537]
    expert_stds = [1.5616618964384956, 1.449172525995787, 1.5205992970609392, 1.5469422429523034, 1.511270787760881, 1.455189251463794]
    expert_taus = sigma_to_tau(expert_stds)

    mu = [3.6, 3.1666666666666665, 3.6, 3.1333333333333333, 3.6666666666666665, 2.3]
    sigma = [1.3544307876819288, 1.2617266038997932, 1.3796551293211172, 1.2521246311585852, 1.5161960871578068, 0.9523111632886272]

    init = tuple(create_init([0]+(mu*num_projects), [0]+(sigma*num_projects)))
    expert_costs = [0.002]*6

    config = MouselabConfig(num_projects, num_criteria, expert_costs, expert_taus, init, criteria_scale=weights, term_belief=True, max_actions=5)
    env = MouselabJas(config=config)
    return env, config