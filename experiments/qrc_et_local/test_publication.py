"""Recompute all public metrics and enforce the two-method comparison contract."""
from pathlib import Path
import unittest

from prepare_publication import digest, validate_export
from publish_wandb import content


class PublicationTests(unittest.TestCase):
    def test_public_data_and_prefixes(self):
        evidence = validate_export(Path(__file__).resolve().parent)
        self.assertEqual(len(evidence["runs"]), 12)
        self.assertEqual(sum(r["audit"]["rolling_returns_reconciled"] for r in evidence["runs"]), 16000)

    def test_agent_snapshot_matches_tested_candidate(self):
        root = Path(__file__).resolve().parents[2]
        self.assertEqual(digest(root / "algorithms/qrc-et-algo3-lab.py"),
                         "1023f4ed6b63ab91ea8d97df617523be07ada970edde7ee27b029286aec759c6")

    def test_report_contains_every_game_and_no_third_method(self):
        evidence = validate_export(Path(__file__).resolve().parent)
        parts = content(evidence, "https://github.com/mubashir20023/eligibility-traces")
        plots = [p for p in parts if p["kind"] == "pair"]
        self.assertEqual(len(plots), 7)
        text = str(parts)
        for game in ["Asterix", "Breakout", "Freeway", "Seaquest", "SpaceInvaders"]:
            self.assertIn(game, text)
        self.assertNotIn("et_original", text)
        self.assertIn("not yet a faithful reproduction", text)


if __name__ == "__main__":
    unittest.main()
