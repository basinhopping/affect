"""The IEMOCAP plan and after-copy checks on small hand-made cases with known answers."""
import unittest

from affect.data.iemocap import after_copy_checks, plan_checks


def rec(uid: str) -> dict:
    return {"utt_id": uid}


class TestPlanChecks(unittest.TestCase):
    def test_all_good(self):
        records = [rec("Ses05M_x_M000"), rec("Ses05M_x_F000")]
        sizes = {"Ses05M_x_M000": 0, "Ses05M_x_F000": 100}
        fill = [{"path": "Ses05M_x/Ses05M_x_M000.wav", "bytes": 90, "sha256": "a"},
                {"path": "Ses05M_x/Ses05M_x_F000.wav", "bytes": 100, "sha256": "b"}]
        r = plan_checks(records, sizes, fill)
        self.assertTrue(r["ok"], r["checks"])
        self.assertEqual(r["n_release_sentence_wavs_empty"], 1)

    def test_each_problem_is_named(self):
        records = [rec("A_M000"), rec("B_M000"), rec("C_F000")]
        sizes = {"A_M000": 0, "B_M000": 10, "Z_F000": 5}          # C has no wav; Z has no label
        fill = [{"path": "Q/Q_M000.wav", "bytes": 5, "sha256": "q"},  # no release counterpart
                {"path": "notes.txt", "bytes": 1, "sha256": "n"}]
        r = plan_checks(records, sizes, fill)
        self.assertFalse(r["ok"])
        failed = {k for k, v in r["checks"].items() if not v}
        self.assertEqual(failed, {"every_release_sentence_wav_is_labeled",
                                  "every_labeled_utterance_has_a_release_wav_file",
                                  "every_empty_release_wav_has_a_collaborator_copy",
                                  "every_collaborator_file_has_a_release_counterpart",
                                  "collaborator_folder_holds_only_wav_files"})


class TestAfterCopyChecks(unittest.TestCase):
    def setUp(self):
        self.fill = {"rows": [{"path": "D/D_M000.wav", "sha256": "fill"},
                              {"path": "D/D_F000.wav", "sha256": "same"}]}
        self.release = {"rows": [{"path": "Session5/sentences/wav/D/D_F000.wav", "sha256": "same"},
                                 {"path": "Session5/dialog/EmoEvaluation/D.txt", "sha256": "lab"}]}
        self.rows = [
            {"path": "audio/Session5/D/D_M000.wav", "source": "collaborator_wav", "source_path": "D/D_M000.wav",
             "sha256": "fill"},
            {"path": "audio/Session5/D/D_F000.wav", "source": "IEMOCAP_full_release",
             "source_path": "Session5/sentences/wav/D/D_F000.wav", "sha256": "same"},
            {"path": "labels/Session5/D.txt", "source": "IEMOCAP_full_release",
             "source_path": "Session5/dialog/EmoEvaluation/D.txt", "sha256": "lab"},
        ]

    def test_all_good(self):
        r = after_copy_checks(self.rows, {"collaborator_wav": self.fill, "IEMOCAP_full_release": self.release})
        self.assertTrue(r["ok"], r["checks"])
        self.assertEqual(r["n_collaborator_files_identical_to_release"], 1)
        self.assertEqual(len(r["checks"]), 3)

    def test_collaborator_copy_that_differs_is_caught(self):
        self.fill["rows"][1]["sha256"] = "other"
        r = after_copy_checks(self.rows, {"collaborator_wav": self.fill})
        self.assertFalse(r["checks"]["collaborator_copies_of_non_empty_release_files_are_identical"])

    def test_release_copy_that_differs_from_its_ingest_is_caught(self):
        self.release["rows"][1]["sha256"] = "changed"
        r = after_copy_checks(self.rows, {"collaborator_wav": self.fill, "IEMOCAP_full_release": self.release})
        self.assertFalse(r["checks"]["release_copies_match_the_release_manifest"])

    def test_without_manifests_nothing_is_claimed(self):
        r = after_copy_checks(self.rows, {})
        self.assertEqual(r["checks"], {})
        self.assertEqual(r["manifests_available"], [])


if __name__ == "__main__":
    unittest.main()
