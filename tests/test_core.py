import unittest

from pit_core import ContractCore, ContractProfile, Sample


PROFILE = ContractProfile.from_dict(
    {
        "value_bounds": {"linear_x": [-0.01, 0.6], "angular_z": [-1.1, 1.1]},
        "max_staleness_ms": 80,
        "compute_budget_ms": 40,
        "transport_budget_ms": 80,
    }
)


def sample(**changes):
    values = {
        "values": {"linear_x": 0.2, "angular_z": 0.1},
        "produced_ns": 0,
        "observed_ns": 20_000_000,
        "compute_ms": 10.0,
        "transport_ms": 1.0,
    }
    values.update(changes)
    return Sample(**values)


class ContractCoreTest(unittest.TestCase):
    def test_normal_is_admitted(self):
        decision = ContractCore(PROFILE, composed=True).evaluate(sample())
        self.assertTrue(decision.admit)
        self.assertFalse(decision.transfer)

    def test_value_violation_requests_safe_replacement(self):
        decision = ContractCore(PROFILE, composed=True).evaluate(
            sample(values={"linear_x": 9.0, "angular_z": 0.1})
        )
        self.assertTrue(decision.admit)
        self.assertTrue(decision.replace_with_safe_value)
        self.assertIn("projection_value:linear_x", decision.violations)

    def test_composed_stale_blocks_and_transfers_once(self):
        core = ContractCore(PROFILE, composed=True)
        first = core.evaluate(sample(observed_ns=81_000_000))
        second = core.evaluate(sample(observed_ns=82_000_000))
        self.assertFalse(first.admit)
        self.assertTrue(first.transfer)
        self.assertFalse(second.admit)
        self.assertFalse(second.transfer)

    def test_independent_exposes_safe_but_late(self):
        decision = ContractCore(PROFILE, composed=False).evaluate(
            sample(observed_ns=121_000_000, compute_ms=120.0)
        )
        self.assertTrue(decision.admit)
        self.assertFalse(decision.transfer)
        self.assertIn("projection_freshness", decision.violations)

    def test_missing_and_nonfinite_fail_closed(self):
        core = ContractCore(PROFILE, composed=True)
        missing = core.evaluate(sample(values={"linear_x": 0.1}))
        nonfinite = core.evaluate(sample(values={"linear_x": float("nan"), "angular_z": 0.0}))
        self.assertFalse(missing.admit)
        self.assertFalse(nonfinite.admit)


if __name__ == "__main__":
    unittest.main()
