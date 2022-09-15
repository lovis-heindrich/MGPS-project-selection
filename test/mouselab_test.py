import unittest
import numpy as np
from src.utils.mouselab_standalone import MouselabJas
from src.utils.data_classes import MouselabConfig
from src.utils.distributions import Normal

class TestEnvCreation(unittest.TestCase):
    def setUp(self) -> None:
        self.expert_costs = [1, 0.5, 2]
        self.expert_taus = [1, 0.01, 0.01]
        self.config = MouselabConfig()
        self.tree = [[1, 2], [], []]
        self.init = [Normal(0, 1), Normal(0, 20), Normal(0, 20)]

    # Create env with ground truth
    def test_ground_truth(self):
        ground_truth = np.array([1, 1.5, 2.5])
        config = MouselabConfig(ground_truth=ground_truth)
        env = MouselabJas(self.tree, self.init, self.expert_costs, self.expert_taus, config)
        self.assertTrue((env.ground_truth[1:]==ground_truth[1:]).all())
        self.assertEqual(env.ground_truth[0], 0)

    # Create env without ground truth
    def test_random_env(self):
        env = MouselabJas(self.tree, self.init, self.expert_costs, self.expert_taus, self.config)
        self.assertEqual(len(self.tree), len(env.ground_truth))
        self.assertEqual(env.expert_truths.shape, (len(self.expert_costs), len(self.init)))

    # Create env with mismatching values


if __name__ == '__main__':
    unittest.main()