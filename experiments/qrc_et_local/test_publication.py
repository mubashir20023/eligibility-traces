"""Recompute all public metrics and enforce the two-method comparison contract."""
from pathlib import Path
import unittest

from prepare_publication import digest, validate_export
from publish_wandb import content
from report_tables import split_tables, table_html


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

    def test_seed_tables_include_time_and_losses(self):
        evidence = validate_export(Path(__file__).resolve().parent)
        parts = split_tables(content(evidence, "https://github.com/mubashir20023/eligibility-traces"))
        tables = [p for p in parts if p["kind"] == "table"]
        overview = next(p for p in tables if p["headers"][:2] == ["Game", "Seed"])
        self.assertEqual(len(overview["rows"]), 10)
        self.assertEqual({row[1] for row in overview["rows"]}, {"601"})
        for header in ["Score", "Curve avg.", "Time (min)", "TD squared", "Trace loss"]:
            self.assertIn(header, overview["headers"])
        individual = [p for p in tables if p["headers"][:2] == ["Seed", "Method"]]
        self.assertEqual(len(individual), 6)
        self.assertTrue(all(len(p["rows"]) == 2 for p in individual))
        self.assertEqual(len({p["key"] for p in tables}), len(tables))
        self.assertIn("combined mean therefore equals", str(parts))
        for part in tables:
            rendered = table_html(part)
            self.assertIn("<thead>", rendered)
            self.assertIn("border-bottom", rendered)
            self.assertNotIn("<script", rendered)


if __name__ == "__main__":
    unittest.main()
