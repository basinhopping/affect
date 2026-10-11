"""Breakdowns by speaker gender and kind, and the gender-routed system (affect.analysis.groups)."""
import unittest

from affect.analysis.groups import group_ua, route_by_gender, speaker_gender

CLASSES = ("neu", "hap+exc", "ang", "sad")


def row(uid, speaker, kind, label, pred):
    return {"utt_id": uid, "speaker": speaker, "kind": kind, "label": label, "pred": pred}


class TestGroups(unittest.TestCase):
    def test_speaker_gender(self):
        self.assertEqual(speaker_gender("Ses03_F"), "F")
        self.assertEqual(speaker_gender("Ses05_M"), "M")
        for bad in ("Ses06_F", "Ses01F", "speaker"):
            with self.assertRaises(ValueError):
                speaker_gender(bad)

    def test_group_ua_known_answer(self):
        rows = []
        for i, c in enumerate(CLASSES):
            rows.append(row(f"f{i}", "Ses01_F", "impro", c, c))                       # female: all right
            rows.append(row(f"m{i}", "Ses01_M", "script", c, CLASSES[(i + 1) % 4]))   # male: all wrong
        rows.append(row("m9", "Ses01_M", "script", "neu", "neu"))                      # male: 1 of 2 neu right
        self.assertEqual(group_ua(rows, "gender", CLASSES), {"F": 1.0, "M": 0.125})
        self.assertEqual(group_ua(rows, "kind", CLASSES), {"impro": 1.0, "script": 0.125})
        missing_class = [r for r in rows if not (r["speaker"] == "Ses01_M" and r["label"] == "sad")]
        self.assertIsNone(group_ua(missing_class, "gender", CLASSES)["M"])
        with self.assertRaises(ValueError):
            group_ua(rows, "session", CLASSES)

    def test_route_by_gender_takes_each_speakers_own_model(self):
        female_model = [row("a", "Ses01_F", "impro", "neu", "neu"), row("b", "Ses01_M", "impro", "ang", "sad")]
        male_model = [row("a", "Ses01_F", "impro", "neu", "sad"), row("b", "Ses01_M", "impro", "ang", "ang")]
        routed = {r["utt_id"]: r["pred"] for r in route_by_gender(female_model, male_model)}
        self.assertEqual(routed, {"a": "neu", "b": "ang"})

    def test_route_by_gender_refuses_mismatched_predictions(self):
        f = [row("a", "Ses01_F", "impro", "neu", "neu")]
        with self.assertRaises(ValueError):
            route_by_gender(f, [row("b", "Ses01_M", "impro", "neu", "neu")])
        with self.assertRaises(ValueError):
            route_by_gender(f, [row("a", "Ses01_F", "impro", "sad", "neu")])
        with self.assertRaises(ValueError):
            route_by_gender(f + f, f)


if __name__ == "__main__":
    unittest.main()
