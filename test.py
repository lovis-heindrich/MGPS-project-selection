from src.utils.mouselab_JAS import MouselabJas
from src.utils.distributions import Normal

tree = [[1, 2], [], []]
init = [Normal(0, 1), Normal(0, 20), Normal(0, 20)]

expert_costs = [1, 0.5, 2]
expert_taus = [1, 0.01, 0.01]

env = MouselabJas(tree, init, expert_costs=expert_costs, expert_taus=expert_taus)
print(env.expert_sigma)
print(env.ground_truth)
print(env.expert_truths)

print(list(env.actions(env._state)))

print(env._step((1,2)))
print(env._step((2,2)))