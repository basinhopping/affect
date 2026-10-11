"""The real assembled IEMOCAP copy, where present (PC after assembly; node after transfer).

Known answers from the Phase 0 audit (_phase0_audit/iemocap_audit.json, 2026-09-25):
10,039 labeled utterances, 5,531 in the four merged classes, 501 audio files that were empty
in Harrison's copy and come from the collaborator's folder.
"""
import unittest

from tests.support import needs_assembled


@needs_assembled("iemocap")
class TestAssembledIEMOCAP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from affect import paths
        from affect.data.iemocap import check_assembled_copy, read_assembly_record
        from affect.data.manifest import read_manifest

        root = paths.assembled_dir("iemocap")
        cls.record = read_assembly_record(root)
        cls.report = check_assembled_copy(root, read_manifest(root / "MANIFEST.csv"))

    def test_checks_pass(self):
        self.assertTrue(self.report["ok"], {k: v for k, v in self.report["checks"].items() if not v})

    def test_known_counts(self):
        self.assertEqual(self.report["n_utterances"], 10039)
        self.assertEqual(self.report["four_class"]["total"], 5531)
        self.assertEqual(self.report["four_class"]["session_totals"],
                         {"S1": 1085, "S2": 1023, "S3": 1151, "S4": 1031, "S5": 1241})
        self.assertEqual(self.report["n_filled_from_collaborator"], 501)
        self.assertEqual(self.report["parity"]["n_compared"], 10039)


if __name__ == "__main__":
    unittest.main()
