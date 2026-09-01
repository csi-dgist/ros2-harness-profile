import unittest

try:  # not part of the public release
    from scripts.build_rebuttal_report import audit_application
except ImportError:
    audit_application = None


@unittest.skipIf(audit_application is None,
                 "scripts/build_rebuttal_report.py is not part of the public release")
class RebuttalReportTest(unittest.TestCase):
    @staticmethod
    def analysis(application="nav2", pit_stale=0):
        stale = "stale_final_runs" if application == "nav2" else "stale_admission_runs"
        fallback = "fallback_evidence_runs" if application == "nav2" else "fallback_runs"
        success = "goal_success" if application == "nav2" else "target_reached"
        independent = {
            "expected": 10, "runs": 10, stale: 10, "transfer_runs": 0,
            fallback: 0, success: 10,
        }
        pit = {
            "expected": 10, "runs": 10, stale: pit_stale, "transfer_runs": 10,
            fallback: 10, success: 10,
        }
        return {
            "matrix_complete": True,
            "invariants_passed": pit_stale == 0,
            "scheduled_runs": 20,
            "cells": {
                "rmw|source|fallback|independent|safe_but_late": independent,
                "rmw|source|fallback|pit|safe_but_late": pit,
            },
        }

    def test_valid_nav2_priority_pack(self):
        summary, errors = audit_application("Nav2", self.analysis(), "nav2")
        self.assertEqual(errors, [])
        self.assertEqual(summary["independent_stale_runs"], 10)
        self.assertEqual(summary["pit_stale_runs"], 0)

    def test_stale_pit_run_fails_closed(self):
        _, errors = audit_application("Nav2", self.analysis(pit_stale=1), "nav2")
        self.assertTrue(any("PIT admitted stale" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
