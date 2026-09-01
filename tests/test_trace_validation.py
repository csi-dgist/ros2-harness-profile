import json
from pathlib import Path
import tempfile
import unittest

try:  # not part of the public release
    from scripts.collect_expert_traces import valid_trace
except ImportError:
    valid_trace = None


@unittest.skipIf(valid_trace is None,
                 "scripts/collect_expert_traces.py is not part of the public release")
class TraceValidationTest(unittest.TestCase):
    def test_trace_requires_revision_sequence_finite_values_and_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "expert_01_trace.json"
            task = root / "expert_01_task.json"
            rows = [
                {
                    "revision": "pit-portability-trace-recorder-v1",
                    "run_id": "expert_01",
                    "sequence": index,
                    "received_wall_ns": 2_000_000 + index,
                    "source_stamp_ns": 1_000_000 + index,
                    "linear_x": 0.1,
                    "angular_z": -0.02,
                }
                for index in range(100)
            ]
            trace.write_text(json.dumps(rows), encoding="utf-8")
            task.write_text(json.dumps({"goal_result": "SUCCEEDED"}), encoding="utf-8")
            self.assertTrue(valid_trace(trace, task))
            rows[20]["sequence"] = 99
            trace.write_text(json.dumps(rows), encoding="utf-8")
            self.assertFalse(valid_trace(trace, task))


if __name__ == "__main__":
    unittest.main()
