import unittest

import numpy as np

from training.train_bc import group_split, mse, train_linear, train_mlp


class TrainingTest(unittest.TestCase):
    def test_linear_recovers_deterministic_mapping(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(200, 8))
        expected_w = rng.normal(size=(8, 2))
        expected_b = np.asarray([0.2, -0.1])
        y = x @ expected_w + expected_b
        w, b = train_linear(x, y, ridge=1e-10)
        self.assertLess(mse(x @ w + b, y), 1e-16)

    def test_mlp_training_reduces_loss(self):
        rng = np.random.default_rng(9)
        x = rng.normal(size=(160, 8))
        y = np.column_stack([np.tanh(x[:, 0] - x[:, 1]), 0.3 * x[:, 2]])
        params, losses = train_mlp(
            x, y, seed=256, hidden=12, epochs=150, learning_rate=0.01
        )
        prediction = np.tanh(x @ params["w1"] + params["b1"]) @ params["w2"] + params["b2"]
        self.assertLess(mse(prediction, y), losses[0] * 0.2)

    def test_group_split_holds_out_complete_expert_run(self):
        groups = np.asarray([f"expert_{run:02d}" for run in range(1, 6) for _ in range(20)])
        train, test, train_runs, test_runs = group_split(groups, seed=256)
        self.assertFalse(train_runs & test_runs)
        self.assertEqual(len(train_runs), 4)
        self.assertEqual(len(test_runs), 1)
        self.assertEqual(set(groups[train]), train_runs)
        self.assertEqual(set(groups[test]), test_runs)


if __name__ == "__main__":
    unittest.main()
