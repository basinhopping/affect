"""The architecture descriptions (affect.analysis.architectures, added 2026-10-26): every record's configuration is found
in its sweep file with its full hash and every screen's base once (no torch); where torch is installed (the node), each
of the eleven networks is built, runs on 3 s of input and gives one logit per class, every leaf layer has its shapes,
and the parameters of the top-level parts add up to the total."""
import unittest

from affect import catalog
from affect.analysis import architectures
from tests.support import needs_module

catalog.load_all()


class TestConfigurations(unittest.TestCase):
    def test_records_then_bases(self):
        found = architectures.configurations()
        self.assertEqual(len(found), len(architectures.RECORDS) + len(architectures.SCREENS))
        self.assertEqual([c[2] for c in found[:len(architectures.RECORDS)]], [r[2] for r in architectures.RECORDS])
        self.assertEqual([c[1] for c in found[len(architectures.RECORDS):]], list(architectures.SCREENS))
        models = [c[3]["model"]["name"] for c in found]
        self.assertEqual(models, ["cnn", "cnn", "cnn", "cnn_transformer", "cnn_lif", "snn", "cnn_rnn", "frames_pool",
                                  "frames_transformer", "frames_lif", "frames_rnn"])


@needs_module("torch")
class TestDescribe(unittest.TestCase):
    def test_every_network_builds_and_runs(self):
        for name, sweep_id, config, settings in architectures.configurations():
            with self.subTest(name=name):
                d = architectures.describe(name, sweep_id, config, settings)
                self.assertEqual(d["logits"], [1, d["n_classes"]])
                self.assertGreater(d["parameters"], 0)
                self.assertEqual(sum(d["parameters_by_part"].values()), d["parameters"])
                called = [r for r in d["layers"] if r["calls"]]
                self.assertTrue(called)
                self.assertTrue(all("output" in r for r in called))
                self.assertIn("(", d["module_tree"])
                text = architectures.render(d)
                self.assertIn(config, text)


if __name__ == "__main__":
    unittest.main()
