import unittest
from src.policy.jas_voc_policy import JAS_voc_policy
from src.utils.mouselab_jas import MouselabJas
from src.utils.data_classes import Action, MouselabConfig
from src.utils.distributions import Normal
from simulation import run_episode, run_simulation
import numpy as np

class TestVOC(unittest.TestCase):
    def setUp(self) -> None:
        config = MouselabConfig(
            num_projects=2,
            num_criterias=1,
            expert_costs=[0.5, 2],
            expert_taus=[0.01, 0.01],
            init=(Normal(0, 1), Normal(0, 20), Normal(0, 10))
        )
        self.env = MouselabJas(config)
        self.policy = JAS_voc_policy()
    
    def test_voc_values(self):
        voc1 = self.policy.myopic_voc_normal(self.env, Action(0, 1))
        voc2 = self.policy.myopic_voc_normal(self.env, Action(0, 2))
        self.assertGreater(voc1, 0)
        self.assertGreater(voc2, 0)
        self.assertGreater(voc1, voc2)

    def test_path_value(self):
        path, alternative = self.policy.get_best_path_action_set(
            self.env, Action(0, 1), self.env.state
        )
        self.assertEqual(path, 0)
        self.assertEqual(alternative, 0)
        self.env.state = (Normal(0, 1), Normal(1, 20), Normal(0, 10))
        path, alternative = self.policy.get_best_path_action_set(
            self.env, Action(0, 1), self.env.state
        )
        self.assertEqual(path, 1)
        self.assertEqual(alternative, 0)
        self.env.state = (Normal(0, 1), Normal(1, 20), Normal(2, 10))
        path, alternative = self.policy.get_best_path_action_set(
            self.env, Action(0, 1), self.env.state
        )
        self.assertEqual(path, 1)
        self.assertEqual(alternative, 2)

    def test_voc_equal_cost(self):
        # Higher precision -> higher voc
        self.env.expert_costs = [1, 1]
        self.env.expert_taus = [2, 1]
        self.env.reset()
        self.assertGreater(self.policy.myopic_voc_normal(self.env, Action(0, 1)), self.policy.myopic_voc_normal(self.env, Action(1, 1)))

    def test_act_equal_tau(self):
        action = self.policy.act(self.env)
        self.assertEqual(action, Action(0, 1))
    
    def test_act_equal_costs(self):
        self.env.expert_costs = [1, 1]
        self.env.expert_taus = [1, 2]
        self.env.reset()
        self.assertEqual(self.policy.act(self.env), Action(1, 1))

class TestSimulationVOC(unittest.TestCase):
    def setUp(self) -> None:
        config = MouselabConfig(
            num_projects=2,
            num_criterias=1,
            expert_costs=[0.01, 0.01],
            expert_taus = [0.001, 0.001],
            init=(Normal(0, 1), Normal(0, 20), Normal(0, 10))
        )
        self.env = MouselabJas(config)
        self.policy = JAS_voc_policy()
    
    def test_seeded_episode(self):
        res1 = run_episode(self.env, self.policy, seed=1)
        res2 = run_episode(self.env, self.policy, seed=1)
        res3 = run_episode(self.env, self.policy, seed=2)
        self.assertTrue(res1.reward == res2.reward != res3.reward)
        self.assertTrue(res1.seed == res2.seed != res3.seed)
        #self.assertTrue(res1.actions == res2.actions != res3.actions)
    
    def test_non_seeded_episode(self):
        res1 = run_episode(self.env, self.policy)
        res2 = run_episode(self.env, self.policy)
        res3 = run_episode(self.env, self.policy)
        self.assertTrue(res1.reward != res2.reward != res3.reward)
        #self.assertTrue(res1.actions != res2.actions != res3.actions)
    
    def test_seeded_evaluation(self):
        res1 = run_simulation(self.env, self.policy, n=10, start_seed=5)
        res2 = run_simulation(self.env, self.policy, n=10, start_seed=5)
        res3 = run_simulation(self.env, self.policy, n=10, start_seed=10)
        self.assertTrue(np.all([i in res1["seed"].tolist() for i in range(5, 15)]))
        self.assertTrue(np.all(res1["reward"] == res2["reward"]))
        self.assertTrue(np.any(res1["reward"] != res3["reward"]))
    
    def test_non_seeded_evaluation(self):
        res = run_simulation(self.env, self.policy, n=10)
        self.assertEqual(len(res), 10)

if __name__ == "__main__":
    unittest.main()
