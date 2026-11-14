import tempfile
import unittest
from pathlib import Path

from drift_watch import compare, jensen_shannon, load_csv, population_stability_index, reference_edges


class DriftWatchTests(unittest.TestCase):
    def test_identical_numeric_distribution_has_zero_psi(self):
        psi, _ = population_stability_index([1, 2, 3, 4], [1, 2, 3, 4])
        self.assertAlmostEqual(psi, 0)

    def test_psi_edges_come_from_reference_only(self):
        psi, edges = population_stability_index([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [100] * 10)
        self.assertTrue(all(edge < 10 for edge in edges))
        self.assertGreater(psi, .2)

    def test_constant_reference_still_handles_new_values(self):
        psi, edges = population_stability_index([2] * 10, [100] * 10)
        self.assertEqual(len(edges), 2)
        self.assertTrue(psi > 0)

    def test_missing_numeric_values_are_counted_separately(self):
        ref = [{"x": 1.0}, {"x": 2.0}, {"x": 3.0}, {"x": 4.0}]
        cur = [{"x": 1.0}, {"x": None}, {"x": None}, {"x": None}]
        result = compare(ref, cur, ["x"], [], missing_delta_threshold=.5)
        self.assertAlmostEqual(result["numeric"]["x"]["current_missing_rate"], .75)
        self.assertTrue(result["numeric"]["x"]["alert"])

    def test_categorical_identical_has_zero_js(self):
        self.assertAlmostEqual(jensen_shannon(["a", "b"], ["a", "b"]), 0)

    def test_categorical_shift_alerts(self):
        ref = [{"c": "web"} for _ in range(9)] + [{"c": "store"}]
        cur = [{"c": "store"} for _ in range(9)] + [{"c": "web"}]
        result = compare(ref, cur, [], ["c"])
        self.assertEqual(result["status"], "drift_alert")

    def test_missing_only_feature_does_not_fabricate_distribution(self):
        result = compare([{"x": None}], [{"x": None}], ["x"], [])
        self.assertIsNone(result["numeric"]["x"]["psi"])

    def test_loader_rejects_bad_numbers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.csv"
            path.write_text("x,c\nNaN,a\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fini"):
                load_csv(path, ["x"], ["c"])
            path.write_text("x,c\noops,a\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "numérique"):
                load_csv(path, ["x"], ["c"])

    def test_duplicate_variable_selection_rejected(self):
        with self.assertRaisesRegex(ValueError, "sans doublons"):
            compare([{"x": 1}], [{"x": 1}], ["x", "x"], [])

    def test_empty_batches_rejected(self):
        with self.assertRaisesRegex(ValueError, "au moins une ligne"):
            compare([], [{"x": 1}], ["x"], [])


if __name__ == "__main__":
    unittest.main()
