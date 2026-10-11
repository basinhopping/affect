"""Statistics against known answers, including a pairing check a wrong bootstrap would fail."""
import math
import unittest

from tests.support import needs_module


@needs_module("scipy")
class TestTInterval(unittest.TestCase):
    def test_five_seeds(self):
        from affect.eval.stats import t_interval

        r = t_interval([1, 2, 3, 4, 5], 0.95)
        t975_df4 = 2.7764451051977987
        self.assertAlmostEqual(r["t_critical"], t975_df4, places=9)
        self.assertAlmostEqual(r["sd"], math.sqrt(2.5), places=12)
        self.assertAlmostEqual(r["half_width"], t975_df4 * math.sqrt(2.5) / math.sqrt(5), places=9)
        # the plan's rule of thumb: half-width = 1.24 x SD for 5 seeds
        self.assertAlmostEqual(r["half_width"] / r["sd"], 1.2416, places=3)


@needs_module("numpy")
class TestHolmAndPOO(unittest.TestCase):
    def test_holm_known_answer(self):
        from affect.eval.stats import holm

        # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max
        adjusted = holm([0.01, 0.04, 0.03, 0.005])
        for got, want in zip(adjusted, [0.03, 0.06, 0.06, 0.02]):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(holm([0.9, 0.8]), [1.0, 1.0])

    def test_probability_of_outperforming(self):
        from affect.eval.stats import probability_of_outperforming

        self.assertAlmostEqual(probability_of_outperforming([1, 2, 3], [2, 2, 2]), 0.5)
        self.assertAlmostEqual(probability_of_outperforming([3, 4], [1, 2]), 1.0)
        self.assertAlmostEqual(probability_of_outperforming([1], [1]), 0.5)


@needs_module("numpy")
class TestPairedBootstrap(unittest.TestCase):
    def make(self):
        import numpy as np

        rng = np.random.default_rng(0)
        folds_a, folds_b = [], []
        for fold in range(2):
            y = np.repeat(np.arange(4), 100)
            base = rng.random(400) < 0.6                     # B right on about 60%
            wrong_class0 = np.flatnonzero((y == 0) & ~base)[:20]
            better = base.copy()
            better[wrong_class0] = True                      # A also right on 20 more class-0 items
            ids = [f"f{fold}_{i}" for i in range(400)]
            folds_b.append({"utt_ids": ids, "y_true": y, "correct": np.tile(base, (3, 1))})
            folds_a.append({"utt_ids": ids, "y_true": y, "correct": np.tile(better, (3, 1))})
        return folds_a, folds_b

    def test_observed_difference_by_hand(self):
        import numpy as np

        from affect.eval.stats import paired_bootstrap_ua_difference

        a, b = self.make()
        r = paired_bootstrap_ua_difference(a, b, 4, 2000, 0.95, np.random.default_rng(1), 500)
        # 20 more right of 100 class-0 items: recall +0.2, UA +0.05 in each fold
        self.assertAlmostEqual(r["difference"], 0.05, places=12)
        self.assertLess(r["low"], 0.05)
        self.assertGreater(r["high"], 0.05)
        self.assertGreater(r["low"], 0.0)

    def test_pairing_narrows_the_interval(self):
        import numpy as np

        from affect.eval.stats import _fold_ua_per_seed, paired_bootstrap_ua_difference

        a, b = self.make()
        paired = paired_bootstrap_ua_difference(a, b, 4, 2000, 0.95, np.random.default_rng(2), 500)
        # An unpaired bootstrap: separate utterance resamples for each model.
        rng = np.random.default_rng(3)
        diffs = []
        for _ in range(2000):
            ua = [0.0, 0.0]
            for fa, fb in zip(a, b):
                onehot = np.eye(4)[fa["y_true"]]
                for k, f in enumerate((fa, fb)):
                    w = rng.multinomial(400, np.full(400, 1 / 400), size=1).astype(float)
                    ua[k] += _fold_ua_per_seed(np.asarray(f["correct"]), onehot, w)[0].mean() / 2
            diffs.append(ua[0] - ua[1])
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        self.assertLess(paired["high"] - paired["low"], 0.6 * (hi - lo))

    def test_unmatched_utterances_refused(self):
        import numpy as np

        from affect.eval.stats import paired_bootstrap_ua_difference

        a, b = self.make()
        b[1] = dict(b[1], utt_ids=list(reversed(b[1]["utt_ids"])))
        with self.assertRaisesRegex(ValueError, "cannot be paired"):
            paired_bootstrap_ua_difference(a, b, 4, 10, 0.95, np.random.default_rng(0), 10)


if __name__ == "__main__":
    unittest.main()
