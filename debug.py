# %%
import numpy as np
import pandas as pd
from src.policy.jas_voc_policy import JAS_voc_policy
from src.utils.mouselab_jas import MouselabJas
from src.utils.data_classes import Action, MouselabConfig
from src.utils.distributions import Normal
from src.utils.utils import sigma_to_tau
from src.utils.env_creation import create_tree, create_init
from simulation import run_simulation
from scipy.stats import norm


# %%
num_projects = 2
num_criteria = 1
weights = [1]
expert_stds = [1,1]
expert_taus = sigma_to_tau(np.array(expert_stds))

mu = [3]
sigma = [1]

init = create_init([0]+(mu*num_projects), [0]+(sigma*num_projects))
expert_costs = [0.01]*2

config = MouselabConfig(num_projects, num_criteria, expert_costs, expert_taus, init, criteria_scale=weights, term_belief=True)
env = MouselabJas(config=config)

# %%
init = create_init([0,4,3], [0, 0.000001, 1])
config = MouselabConfig(num_projects, num_criteria, expert_costs, expert_taus, init, criteria_scale=weights, term_belief=True)
env = MouselabJas(config=config)

env.expert_truths = np.array([[1., 4., 5.], [1., 5., 5.]])
env.step(Action(0, 2))
env.step(Action(1, 2))

# %%
policy_disc = JAS_voc_policy(discrete_observations=True)
env = MouselabJas(config=config)
actions = list(env.actions())
print([(action, policy_disc.myopic_voc_normal(env, action)) for action in actions])

# %%
policy_cont = JAS_voc_policy(discrete_observations=False)
env = MouselabJas(config=config)
actions = list(env.actions())
print([(action, policy_cont.myopic_voc_normal(env, action)) for action in actions])

# %%
action = Action(expert=1, query=1)
voc_disc = policy_disc.myopic_voc_normal(env, action)
voc_cont = policy_cont.myopic_voc_normal(env, action)
print(voc_disc, voc_cont)

# %%


# %%
def compare(config, n=100, seed=0):
    env = MouselabJas(config=config)
    voc_policy_disc = JAS_voc_policy(discrete_observations=True)
    voc_policy = JAS_voc_policy(discrete_observations=False)
    res_voc = run_simulation(env, voc_policy, start_seed=seed, n=n)
    res_voc["algorithm"] = "MGPO"
    res_voc_disc = run_simulation(env, voc_policy_disc, start_seed=seed, n=n)
    res_voc_disc["algorithm"] = "MGPO_disc"
    res = pd.concat([res_voc, res_voc_disc])
    return res

# %%
res = compare(config, 1, 2)
res.groupby("algorithm").agg(["mean", "std"])

# %%
# Number of projects:
num_projects = [(40+12+22)/3, 7, 50, 25, 27, 27, 10, 24, 4, 4, 37, (20+50+150+250)/4, 4,2,5,400,2,12, (10+30)/2, 6, (5+30)/2, (1000+200)/2, 3, (5+30)/2, 8, 6, 10,
    4, 5, 20, 14, (60+50)/2, 3, 18, 6, 37, 14, 6, (2+3)/2, 20, 20, 4, 20, 14, 5, 20, 20, 5, 45, 1500, 6, 5, 60, 10, 50, 6, 5, 100, 5, 4]
print("Mean", np.mean(num_projects))
print("Median", np.median(num_projects))

# %%
papers_with_criteria = 66
num_papers = [3, 10, 5, 5, 9, 28, 41, 48, 9, 20, 3, 38, 38, 17, 21, 21, 7, 12, 16, 9, 4, 15, 40]
mean_criteria = np.sum(num_papers)/papers_with_criteria
print("Mean", mean_criteria)
top_criteria = np.argsort(num_papers)+1
print(list(top_criteria)[-10:][::-1])

utilized_in_half = [7,8,12,13,23]

# %%
electric_criteria = [13, 17, 12, 4]
it_criteria = [5, 7, 49, 15, 5]
pharmaceutical_criteria = [4]


# %%
# Domains
domains = ["Eletronic/Electricity and Chemical", "Textile", "Aerospacial", "Spacial", "Chemical", "Chemical", 
    "Eletronic/Electricity", "Industrial", "Non-Specified", "Non-Specified", "Metallurgy", "Eletronic/Electricity",
    "Non-Specified", "Non-Specified", "Biotechnology", "Automotive", "Information Technology", "Industrial", "Industrial"
    "Research", "Pharmaceutical", "Non-Specified", "Metallurgy", "Pharmaceutical", "Non-Specified", "Information Technology",
    "Information Technology", "Telecommunications", "Non-Specified", "Non-Specified", "Nuclear", "Pharmaceutical", "Non-Specified",
    "Non-Specified", "Industrial", "Non-Specified", "Metallurgy", "Eletronic/Electricity", "Non-Specified", "Government Sponsored"
    "Civil, Mechanical, and others", "Non-Specified", "Pharmaceutical", "Pharmaceutical", "Manufacturing", "Non-Specified",
    "Pharmaceutical", "Information Technology", "Civil, Mechanical, and others", "Pharmaceutical", "Non-Specified", "Eletronic/Electricity",
    "Agriculture, Innovation, Textile and others", "Private/Public Sector", "Non-Specified", "Non-Specified", "Eletronic/Electricity",
    "Government Sponsored", "Non-Specified", "Pharmaceutical", "Private/Public Sector", "Energy", "Military", "Innovation",
    "Electricity/Mechanical/Telecommunications/IT", "Non-Specified"]

from collections import Counter

counter = Counter(domains)
counter.most_common()


# %%
def diff_rec(str1, str2):
    if (len(str1) == 0) and (len(str2) == 0):
        return 0
    elif len(str1) == 0:
        return sum([ord(c) for c in str2])
    elif len(str2) == 0: 
        return sum([ord(c) for c in str1])
    elif str1[0] == str2[0]:
        return diff_rec(str1[1:], str2[1:])
    else:
        diff1 = ord(str1[0]) + diff_rec(str1[1:], str2)
        diff2 = ord(str2[0]) + diff_rec(str1, str2[1:])
        return min(diff1, diff2)

print(diff_rec("through", "trouf"))


