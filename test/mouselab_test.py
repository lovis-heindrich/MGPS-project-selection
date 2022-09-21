import unittest
import numpy as np
from src.utils.mouselab_standalone import MouselabJas
from src.utils.data_classes import Action, MouselabConfig
from src.utils.distributions import Normal

class TestEnvCreation(unittest.TestCase):
    def setUp(self) -> None:
        self.expert_costs = [1, 0.5, 2]
        self.expert_taus = [1, 0.01, 0.01]
        self.config = MouselabConfig()
        self.tree = [[1, 2], [], []]
        self.init = [Normal(0, 1), Normal(0, 20), Normal(0, 20)]

    def test_ground_truth(self):
        ground_truth = np.array([1, 1.5, 2.5])
        config = MouselabConfig(ground_truth=ground_truth)
        env = MouselabJas(self.tree, self.init, self.expert_costs, self.expert_taus, config)
        self.assertTrue((env.ground_truth[1:]==ground_truth[1:]).all())
        self.assertEqual(env.ground_truth[0], 0)

    def test_random_env(self):
        env = MouselabJas(self.tree, self.init, self.expert_costs, self.expert_taus, self.config)
        self.assertEqual(len(self.tree), len(env.ground_truth))
        self.assertEqual(env.expert_truths.shape, (len(self.expert_costs), len(self.init)))

class TestReward(unittest.TestCase):
    def setUp(self) -> None:
        self.expert_costs = [1, 0.5, 2]
        self.expert_taus = [1, 0.01, 0.01]
        self.ground_truth = np.array([1, 2.5, 7.5])
        self.tree = [[1], [2], []]
        self.init = [Normal(0, 1), Normal(0, 20), Normal(0, 20)]
    
    def test_expected_term_reward(self):
        config = MouselabConfig(ground_truth=self.ground_truth, term_belief=True)
        env = MouselabJas(self.tree, self.init, self.expert_costs, self.expert_taus, config)
        self.assertEqual(env.term_reward(), env.term_reward(env.state))
        self.assertEqual(env.term_reward(), 0)
        self.assertEqual(env.term_reward(), env.expected_term_reward(env.state))
    
    def test_expected_term_reward_non_zero(self):
        config = MouselabConfig(ground_truth=self.ground_truth, term_belief=True)
        env = MouselabJas(self.tree, self.init, self.expert_costs, self.expert_taus, config)
        env.state[1].mu = 5
        env.state[2].mu = 4
        self.assertEqual(env.term_reward(), env.term_reward(env.state))
        self.assertEqual(env.term_reward(), 9)
        self.assertEqual(env.term_reward(), env.expected_term_reward(env.state))
    
    def test_term_reward(self):
        config = MouselabConfig(ground_truth=self.ground_truth, term_belief=False)
        env = MouselabJas(self.tree, self.init, self.expert_costs, self.expert_taus, config)
        self.assertEqual(env.term_reward(), env.term_reward(env.state))
        self.assertEqual(env.term_reward(), 10)
        self.assertEqual(env.expected_term_reward(env.state), 0)

    def test_sample_term_reward(self):
        ground_truth = np.array([1, 2.5, 7.5])
        tree = [[1, 2], [], []]
        config = MouselabConfig(ground_truth=ground_truth, term_belief=False, sample_term_reward=True)
        env = MouselabJas(tree, self.init, self.expert_costs, self.expert_taus, config)
        self.assertTrue(env.term_reward()==2.5 or env.term_reward()==7.5)
        self.assertEqual(env.expected_term_reward(env.state), 0)
    
    def test_dont_sample_term_reward(self):
        ground_truth = np.array([1, 2.5, 7.5])
        tree = [[1, 2], [], []]
        config = MouselabConfig(ground_truth=ground_truth, term_belief=False, sample_term_reward=False)
        env = MouselabJas(tree, self.init, self.expert_costs, self.expert_taus, config)
        self.assertEqual(env.term_reward(), 5)
        self.assertEqual(env.expected_term_reward(env.state), 0)

class TestAction(unittest.TestCase):
    def setUp(self) -> None:
        expert_costs = [1, -0.5]
        expert_taus = [2, 4]
        ground_truth = np.array([1, 2.5, 7.5])
        tree = [[1], [2], []]
        init = [Normal(0, 1), Normal(0, 4), Normal(0, 4)]
        config = MouselabConfig(ground_truth=ground_truth, limit_repeat_clicks=1)
        self.env = MouselabJas(tree, init, expert_costs, expert_taus, config)
        self.env.expert_truths = np.array([[0, 10, -10], [0, 5, -5]])

    def test_available_actions(self):
        actions = tuple(self.env.actions())
        self.assertEqual(len(actions), 5)

        self.env.step(Action(0,1))
        actions = tuple(self.env.actions())
        self.assertFalse(Action(0,1) in actions)
        self.assertTrue(Action(0,2) in actions)
        self.assertTrue(Action(1,1) in actions)
        self.assertEqual(len(actions), 4)
        self.assertFalse(self.env.done)

        self.env.step(self.env.term_action)
        actions = tuple(self.env.actions())
        self.assertEqual(len(actions), 0)
        self.assertTrue(self.env.done)
    
    def test_action_costs(self):
        _, reward, _, _ = self.env.step(Action(0, 1))
        self.assertEqual(reward, -1)
        _, reward, _, _ = self.env.step(Action(0, 2))
        self.assertEqual(reward, -1)
        _, reward, _, _ = self.env.step(Action(1, 1))
        self.assertEqual(reward, -0.5)
        _, reward, _, _ = self.env.step(Action(1, 2))
        self.assertEqual(reward, -0.5)

    def test_one_step_state_update(self):
        state, _, _, obs = self.env.step(Action(0, 1))
        self.assertEqual(obs, 10)
        prior_mean = 0
        prior_var = 16
        obs_mean = 10
        obs_var = 1 / 2
        post_var = (1 / ((1/prior_var)+(1/obs_var)))
        post_mean = post_var*((prior_mean/prior_var)+(obs_mean/obs_var))
        self.assertAlmostEqual(post_var, state[1].sigma**2)
        self.assertAlmostEqual(post_mean, state[1].mu)
        self.assertEqual(state[1], self.env.state[1])
    
    def test_two_step_state_update(self):
        _, _, _, _ = self.env.step(Action(0, 1))
        state, _, _, obs = self.env.step(Action(1, 1))
        self.assertEqual(obs, 5)

        prior_mean = 0
        prior_var = 16
        obs_mean = 10
        obs_var = 1 / 2
        post_var = (1 / ((1/prior_var)+(1/obs_var)))
        post_mean = post_var*((prior_mean/prior_var)+(obs_mean/obs_var))

        prior_mean = post_mean
        prior_var = post_var
        obs_mean = 5
        obs_var = 1 / 4
        post_var = (1 / ((1/prior_var)+(1/obs_var)))
        post_mean = post_var*((prior_mean/prior_var)+(obs_mean/obs_var))

        self.assertAlmostEqual(post_var, state[1].sigma**2)
        self.assertAlmostEqual(post_mean, state[1].mu)
        self.assertEqual(state[1], self.env.state[1])
        

if __name__ == '__main__':
    unittest.main()