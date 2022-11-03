from src.utils.mouselab_jas import MouselabJas
from src.utils.distributions import Normal
from src.utils.data_classes import MouselabConfig, Action

tree = [[1, 2], [], []]
init = [Normal(0, 1), Normal(0, 20), Normal(0, 20)]

expert_costs = [1, 0.5, 2]
expert_taus = [1, 0.01, 0.01]

config = MouselabConfig([1.1, 2.1, 0.1213], False, False, 1, 5)
env = MouselabJas(tree, init, expert_costs=expert_costs, expert_taus=expert_taus, config=config)
print(env.expert_sigma)
print(env.ground_truth)
print(env.expert_truths)

print(list(env.actions(env.state)))

print(env.step(Action(1,2)))
print(env.step(Action(2,2)))
print(env.step(Action(3,3)))