"""Speed perturbation, crops, evaluation segments, gender labels and mixup draws: the numpy parts,
without torch (added 2026-09-26)."""
import unittest

import numpy as np

from affect.augment import mixup
from affect.augment.settings import AugmentSettings, check
from affect.augment.speed import at_speed, change_speed, draw_speeds, speed_ratio
from affect.data.base import Utterance
from affect.train import data
from tests.support import needs_module


def augment(**changes) -> AugmentSettings:
    base = dict(freq_masks=0, freq_mask_width=0, time_masks=0, time_mask_width=0, speed_factors=(1.0,),
                mixup_alpha=0.0)
    return AugmentSettings(**{**base, **changes})


class TestSpeedSettings(unittest.TestCase):
    def test_ratios(self):
        self.assertEqual(speed_ratio(0.9), (10, 9))
        self.assertEqual(speed_ratio(1.1), (10, 11))
        self.assertEqual(speed_ratio(1.0), (1, 1))
        self.assertEqual(speed_ratio(1.05), (20, 21))
        for bad in (0.0, -1.0, 1 / 3 + 1e-4):
            with self.assertRaises(ValueError):
                speed_ratio(bad)

    def test_check(self):
        check(augment(speed_factors=(0.9, 1.0, 1.1), mixup_alpha=0.2), 64)
        for bad in (dict(speed_factors=()), dict(speed_factors=(1.0, 1.0)), dict(speed_factors=(3.0,)),
                    dict(speed_factors=(1.0 / 3,)), dict(mixup_alpha=-0.1)):
            with self.assertRaises(ValueError, msg=str(bad)):
                check(augment(**bad), 64)


@needs_module("scipy")
class TestSpeed(unittest.TestCase):
    def test_speed_changes_length_and_pitch(self):
        rate = 16000
        t = np.arange(2 * rate) / rate
        x = np.rint(8000 * np.sin(2 * np.pi * 440 * t)).astype(np.int16)
        self.assertIs(change_speed(x, 1.0), x)
        for speed in (0.9, 1.1):
            y = change_speed(x, speed)
            self.assertEqual(y.dtype, np.int16)
            self.assertEqual(y.shape[0], int(np.ceil(x.shape[0] / speed)))
            spectrum = np.abs(np.fft.rfft(y.astype(float) * np.hanning(y.shape[0])))
            peak = np.fft.rfftfreq(y.shape[0], 1 / rate)[spectrum.argmax()]
            self.assertAlmostEqual(peak, 440 * speed, delta=1.0)
            self.assertLess(np.abs(y.astype(float)).max(), 8100)

    def test_resampled_copies_are_kept_and_follow_the_list(self):
        a = [np.arange(1000, dtype=np.int16), np.arange(500, dtype=np.int16)]
        first = at_speed(a, 0, 1.1)
        self.assertIs(at_speed(a, 0, 1.1), first)
        self.assertIs(at_speed(a, 1, 1.0), a[1])
        b = [np.zeros(1000, dtype=np.int16)]
        self.assertEqual(int(np.abs(at_speed(b, 0, 1.1)).max()), 0, "another list gets its own copies")

    def test_training_waves_without_speeds_is_crop_batch_exactly(self):
        rng_audio = np.random.default_rng(3)
        audio = [rng_audio.integers(-3000, 3000, size=n).astype(np.int16) for n in (900, 2000, 3500, 400)]
        rows = np.array([2, 0, 3, 1, 2])
        a, b = np.random.default_rng(5), np.random.default_rng(5)
        spare = np.random.default_rng(9)
        w1, v1 = data.crop_batch(audio, rows, 1000, a)
        w2, v2 = data.training_waves(audio, rows, 1000, (1.0,), b, spare)
        np.testing.assert_array_equal(w1, w2)
        np.testing.assert_array_equal(v1, v2)
        self.assertEqual(a.integers(0, 10 ** 9), b.integers(0, 10 ** 9), "the crop stream is used alike")
        self.assertEqual(spare.integers(0, 10 ** 9), np.random.default_rng(9).integers(0, 10 ** 9),
                         "no speed perturbation draws nothing from the augmentation stream")

    def test_training_waves_with_speeds(self):
        audio = [np.full(n, 1000, dtype=np.int16) for n in (900, 2000, 3500)]
        rows = np.array([0, 1, 2, 0])
        waves, valid = data.training_waves(audio, rows, 1000, (0.9, 1.1), np.random.default_rng(0),
                                           np.random.default_rng(1))
        self.assertEqual(waves.shape, (4, 1000))
        speeds = draw_speeds((0.9, 1.1), 4, np.random.default_rng(1))
        for j, row in enumerate(rows):
            expected = min(1000, int(np.ceil(audio[row].shape[0] / speeds[j])))
            self.assertEqual(valid[j], expected)
            self.assertTrue(np.all(waves[j, valid[j]:] == 0))


class TestSpeedDraws(unittest.TestCase):
    def test_single_factor_draws_nothing(self):
        rng = np.random.default_rng(4)
        np.testing.assert_array_equal(draw_speeds((0.9,), 3, rng), [0.9, 0.9, 0.9])
        self.assertEqual(rng.integers(0, 10 ** 9), np.random.default_rng(4).integers(0, 10 ** 9))

    def test_factors_equally_likely(self):
        draws = draw_speeds((0.9, 1.0, 1.1), 30000, np.random.default_rng(0))
        for f in (0.9, 1.0, 1.1):
            self.assertAlmostEqual(np.mean(draws == f), 1 / 3, delta=0.01)


class TestSegments(unittest.TestCase):
    def test_segment_starts(self):
        self.assertEqual(data.segment_starts(500, 1000, 500), [0])
        self.assertEqual(data.segment_starts(1000, 1000, 500), [0])
        self.assertEqual(data.segment_starts(2000, 1000, 500), [0, 500, 1000])
        self.assertEqual(data.segment_starts(2100, 1000, 500), [0, 500, 1000, 1100])
        self.assertEqual(data.segment_starts(1001, 1000, 1000), [0, 1])
        with self.assertRaises(ValueError):
            data.segment_starts(100, 0, 1)

    def test_segment_batches_cover_every_segment_once(self):
        audio = [np.arange(n, dtype=np.int16) for n in (300, 2100, 1000, 50)]
        rows = np.array([1, 3, 0, 2])
        seen = []
        for positions, waves, lengths in data.segment_batches(audio, rows, 3, 1000, 500):
            self.assertLessEqual(waves.shape[0], 3)
            self.assertEqual(waves.shape[1], lengths.max())
            for j, k in enumerate(positions):
                segment = waves[j, :lengths[j]] * 32768.0
                source = audio[rows[k]].astype(np.float32)
                start = next(s for s in data.segment_starts(source.shape[0], 1000, 500)
                             if np.array_equal(source[s:s + lengths[j]], segment) and (k, s) not in seen)
                seen.append((int(k), start))
                self.assertTrue(np.all(waves[j, lengths[j]:] == 0))
        expected = [(k, s) for k in range(4) for s in data.segment_starts(audio[rows[k]].shape[0], 1000, 500)]
        self.assertEqual(sorted(seen), sorted(expected))


class TestGenderIndex(unittest.TestCase):
    def test_gender_index(self):
        def utt(i, g):
            return Utterance(utt_id=f"u{i}", dataset="d", speaker="s", fold_group="S1", dialog="d", kind="impro",
                             raw_label="neu", label="neu", audio_path="a", n_frames=1, sample_rate=16000,
                             transcript=None, extra={} if g is None else {"speaker_gender": g})

        np.testing.assert_array_equal(data.gender_index([utt(0, "F"), utt(1, "M"), utt(2, "F")]), [0, 1, 0])
        with self.assertRaises(ValueError):
            data.gender_index([utt(0, "F"), utt(1, None)])


class TestGenderReport(unittest.TestCase):
    def test_report_handles_one_gender_splits(self):
        from affect.train.loop import gender_report

        probs = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4], [0.3, 0.7]])
        both = gender_report(np.array([0, 1, 1, 1]), probs)
        self.assertEqual(both["accuracy"], 0.75)
        self.assertEqual(both["recall_F"], 1.0)
        self.assertAlmostEqual(both["recall_M"], 2 / 3)
        self.assertAlmostEqual(both["ua"], (1.0 + 2 / 3) / 2)
        female_only = gender_report(np.array([0, 0, 0, 0]), probs)
        self.assertIsNone(female_only["recall_M"])
        self.assertEqual(female_only["n_M"], 0)
        self.assertEqual(female_only["ua"], female_only["recall_F"])


class TestMixupDraws(unittest.TestCase):
    def test_draws(self):
        weights, partner = mixup.draw(0.4, 1000, np.random.default_rng(0))
        self.assertEqual(weights.dtype, np.float32)
        self.assertTrue(((weights >= 0) & (weights <= 1)).all())
        self.assertEqual(sorted(partner.tolist()), list(range(1000)))
        self.assertAlmostEqual(float(weights.mean()), 0.5, delta=0.05)
        again = mixup.draw(0.4, 1000, np.random.default_rng(0))
        np.testing.assert_array_equal(again[0], weights)
        with self.assertRaises(ValueError):
            mixup.draw(0.0, 3, np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
