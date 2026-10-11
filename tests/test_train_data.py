"""The torch-free parts of training: speakers, crops, evaluation batches, class balance."""
import types
import unittest

from tests.support import needs_module


def utterances(speakers):
    return [types.SimpleNamespace(speaker=s) for s in speakers]


@needs_module("numpy")
class TestTrainData(unittest.TestCase):
    def test_speaker_index(self):
        from affect.train.data import speaker_index

        names, speaker_of = speaker_index(utterances(["Ses02_M", "Ses01_F", "Ses02_M", "Ses01_M"]))
        self.assertEqual(names, ("Ses01_F", "Ses01_M", "Ses02_M"))
        self.assertEqual(speaker_of.tolist(), [2, 0, 2, 1])

    def test_rows_per_speaker_covers_every_speaker(self):
        import numpy as np

        from affect.train.data import rows_per_speaker

        speaker_of = np.array([1, 0, 1, 1, 2, 0, 2, 2, 2])
        rows = rows_per_speaker(speaker_of, 2)
        self.assertEqual(rows.tolist(), [1, 5, 0, 2, 4, 6])
        self.assertEqual(set(speaker_of[rows].tolist()), {0, 1, 2})

    def test_eval_batches_cover_every_row_once_longest_first(self):
        import numpy as np

        from affect.train.data import eval_batches

        audio = [np.full(n, 1000 + n, dtype=np.int16) for n in (5, 9, 3, 7, 4)]
        rows = np.array([4, 0, 3, 1])
        seen = []
        for positions, waves, lengths in eval_batches(audio, rows, 3):
            self.assertEqual(waves.shape, (len(positions), int(lengths.max())))
            for j, k in enumerate(positions):
                n = audio[rows[k]].shape[0]
                self.assertEqual(int(lengths[j]), n)
                self.assertTrue(np.allclose(waves[j, :n], (1000 + n) / 32768.0))
                self.assertTrue((waves[j, n:] == 0).all())
            seen.extend(positions.tolist())
        self.assertEqual(sorted(seen), [0, 1, 2, 3])
        self.assertEqual([audio[rows[k]].shape[0] for k in seen], [9, 7, 5, 4])

    def test_crop_batch(self):
        import numpy as np

        from affect.train.data import crop_batch

        audio = [np.arange(10, dtype=np.int16), np.arange(3, dtype=np.int16)]
        waves, valid = crop_batch(audio, np.array([0, 1]), 6, np.random.default_rng(0))
        self.assertEqual(waves.shape, (2, 6))
        self.assertEqual(valid.tolist(), [6, 3])
        start = int(round(waves[0, 0] * 32768))
        self.assertTrue(np.allclose(waves[0] * 32768, np.arange(start, start + 6)))
        self.assertTrue((waves[1, 3:] == 0).all())

    def test_class_weights_and_balanced_order(self):
        import numpy as np

        from affect.train.data import class_weights, train_order

        labels = np.array([0] * 60 + [1] * 30 + [2] * 10)
        w = class_weights(labels, 3)
        self.assertTrue(np.allclose(w, [100 / 180, 100 / 90, 100 / 30]))
        order = train_order(labels, "balanced_sampler", np.random.default_rng(0))
        share = np.bincount(labels[order], minlength=3) / order.shape[0]
        self.assertLess(np.abs(share - 1 / 3).max(), 0.12)
        self.assertEqual(sorted(train_order(labels, "none", np.random.default_rng(0)).tolist()), list(range(100)))
        with self.assertRaises(ValueError):
            class_weights(np.array([0, 0, 1]), 3)


if __name__ == "__main__":
    unittest.main()
