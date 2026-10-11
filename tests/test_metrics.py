"""Metrics against hand-computed answers (plan 6.6)."""
import unittest

from tests.support import needs_module


@needs_module("numpy")
class TestMetrics(unittest.TestCase):
    def setUp(self):
        from affect.eval import metrics

        self.m = metrics
        # true 0,0,0,1,1,2,2,2,2,3 ; predicted 0,0,1,1,1,2,2,0,3,3
        self.y_true = [0, 0, 0, 1, 1, 2, 2, 2, 2, 3]
        self.y_pred = [0, 0, 1, 1, 1, 2, 2, 0, 3, 3]

    def test_confusion_matrix(self):
        cm = self.m.confusion_matrix(self.y_true, self.y_pred, 4)
        self.assertEqual(cm.tolist(), [[2, 1, 0, 0], [0, 2, 0, 0], [1, 0, 2, 1], [0, 0, 0, 1]])

    def test_ua_wa_f1_by_hand(self):
        r = self.m.report(self.y_true, self.y_pred, ("a", "b", "c", "d"))
        # recalls 2/3, 1, 1/2, 1
        self.assertAlmostEqual(r["ua"], (2 / 3 + 1 + 0.5 + 1) / 4, places=12)
        self.assertAlmostEqual(r["wa"], 0.7, places=12)
        # precisions 2/3, 2/3, 1, 1/2 -> F1 2/3, 4/5, 2/3, 2/3 -> macro 0.7
        self.assertAlmostEqual(r["macro_f1"], 0.7, places=12)
        self.assertEqual(r["support"], {"a": 3, "b": 2, "c": 4, "d": 1})

    def test_class_never_predicted_has_zero_f1(self):
        r = self.m.report([0, 1], [0, 0], ("a", "b"))
        self.assertEqual(r["precision"]["b"], 0.0)
        self.assertEqual(r["f1"]["b"], 0.0)
        self.assertAlmostEqual(r["ua"], 0.5)
        self.assertAlmostEqual(r["macro_f1"], (2 / 3) / 2)

    def test_class_without_examples_raises(self):
        with self.assertRaises(ValueError):
            self.m.report([0, 0], [0, 1], ("a", "b"))

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            self.m.confusion_matrix([0, 1], [0], 2)
        with self.assertRaises(ValueError):
            self.m.confusion_matrix([0, 2], [0, 1], 2)
        with self.assertRaises(ValueError):
            self.m.confusion_matrix([0.0, 1.0], [0, 1], 2)

    def test_group_breakdown(self):
        groups = ["impro"] * 5 + ["script"] * 5
        out = self.m.report_by_group(self.y_true, self.y_pred, groups, ("a", "b", "c", "d"))
        self.assertIsNone(out["impro"]["ua"])          # impro holds no class c or d examples
        self.assertEqual(out["impro"]["support"], {"a": 3, "b": 2, "c": 0, "d": 0})
        self.assertIsNone(out["script"]["ua"])         # script holds no class a or b examples


if __name__ == "__main__":
    unittest.main()
