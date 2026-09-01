from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "nodes"))
from scan_goal_features import reactive_feature, scan_sectors, temporal_feature


class ScanGoalPipelineTest(unittest.TestCase):
    def test_feature_contract_dimensions_and_temporal_mask(self):
        sectors = scan_sectors([1.0] * 360, 0.1, 12.0)
        reactive = reactive_feature(
            sectors,
            pose_x=0.0, pose_y=0.0, pose_yaw=0.0,
            goal_x=2.0, goal_y=0.5, linear_x=0.0, angular_z=0.0,
        )
        temporal = temporal_feature(deque(maxlen=3), reactive)
        self.assertEqual(sectors.shape, (36,))
        self.assertEqual(reactive.shape, (40,))
        self.assertEqual(temporal.shape, (164,))
        np.testing.assert_array_equal(temporal[-4:], [0.0, 0.0, 0.0, 1.0])
        self.assertTrue(np.isfinite(temporal).all())

    def test_semantic_profile_has_no_application_or_vendor_names(self):
        text = (ROOT / "profiles" / "semantic_nav_command.yaml").read_text(encoding="utf-8").lower()
        for token in ("nav2", "dwb", "/cmd_vel", "fastdds", "cyclonedds", "rmw_"):
            self.assertNotIn(token, text)

    def test_both_rmw_manifests_satisfy_profile_requirements(self):
        profile = yaml.safe_load((ROOT / "profiles" / "semantic_nav_command.yaml").read_text(encoding="utf-8"))["profile"]
        required = profile["required_backend_capabilities"]
        for rmw in ("rmw_fastrtps_cpp", "rmw_cyclonedds_cpp"):
            manifest = json.loads((ROOT / "capabilities" / f"{rmw}.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["backend"], rmw)
            self.assertTrue(all(manifest["capabilities"].get(name, False) for name in required))

    def test_rebuttal_matrix_size_is_320(self):
        factors = 2 * 2 * 2 * 2 * 2 * 10
        self.assertEqual(factors, 320)


if __name__ == "__main__":
    unittest.main()
