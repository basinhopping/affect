"""The IEMOCAP label and transcript parsers on hand-written files with known answers. The files follow IEMOCAP's
format; their times, labels, scores, comments and words are made up, not taken from the corpus."""
import tempfile
import unittest
from pathlib import Path

from affect.data.iemocap import parse_label_file, parse_transcript_file

LABELS = """% [START_TIME - END_TIME] TURN_NAME EMOTION [V, A, D]

[3.1000 - 5.5000]\tSes01F_impro01_F000\tneu\t[3.0000, 2.5000, 2.0000]
C-E2:\tNeutral;\t()
C-E3:\tNeutral;\t()
C-E4:\tNeutral; Frustration;\t()
C-F1:\tNeutral;\t(example comment)
A-E3:\tval 4; act 2; dom  3;\t()
A-E4:\tval 2; act 3; dom  2;\t(another example comment)
A-F1:\tval 3; act 2; dom  1;\t()

[7.0000 - 9.2500]\tSes01F_impro01_M000\tang\t[2.0000, 4.5000, 3.5000]
C-E2:\tAnger; Frustration;\t()
C-E3:\tFrustration; Anger;\t()
C-E4:\tAnger;\t()
C-M1:\tAnger;\t()
A-E3:\tval 1; act 4; dom  4;\t()
A-E4:\tval 2; act 4; dom  4;\t()
A-M1:\tval 2; act 5; dom  5;\t()
"""


class TestParse(unittest.TestCase):
    def test_label_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Ses01F_impro01.txt"
            path.write_text(LABELS, encoding="utf-8")
            problems = {}
            recs = parse_label_file(path, problems)
        self.assertEqual(problems, {})
        self.assertEqual([r["utt_id"] for r in recs], ["Ses01F_impro01_F000", "Ses01F_impro01_M000"])
        f, m = recs
        self.assertEqual((f["start"], f["end"], f["header_dur"]), (3.1, 5.5, 2.4))
        self.assertEqual((f["session"], f["speaker"], f["marker_gender"], f["kind"], f["turn"]),
                         (1, "Ses01_F", "F", "impro", "000"))
        self.assertEqual(m["speaker"], "Ses01_M")      # the speaker is the letter in the last field
        self.assertEqual(f["votes_all"], {"fru": 1, "neu": 3})
        self.assertEqual(f["votes_first"], {"neu": 3})
        self.assertEqual(m["votes_all"], {"ang": 3, "fru": 2})
        self.assertEqual(m["votes_first"], {"ang": 2, "fru": 1})
        self.assertEqual((f["n_cat_E"], f["n_dim_E"], f["n_self_cat"], f["n_self_dim"]), (3, 2, 1, 1))
        self.assertEqual(f["dim_E"], [{"eval": "E3", "val": 4, "act": 2, "dom": 3},
                                      {"eval": "E4", "val": 2, "act": 3, "dom": 2}])
        self.assertEqual((m["v"], m["a"], m["d"]), (2.0, 4.5, 3.5))

    def test_malformed_header_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Ses01F_impro01.txt"
            path.write_text("[3.1000 - 5.5000]\tSes01F_impro01_F000\tneu\n", encoding="utf-8")
            problems = {}
            self.assertEqual(parse_label_file(path, problems), [])
        self.assertEqual(len(problems["unparsed_header_lines"]), 1)

    def test_transcripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.txt"
            path.write_text("Ses01F_impro01_F000 [003.1000-005.5000]: Hello there.\nM: Okay.\n", encoding="utf-8")
            problems = {}
            out = parse_transcript_file(path, problems)
        self.assertEqual(out, {"Ses01F_impro01_F000": "Hello there."})
        self.assertEqual(len(problems["transcript_lines_unparsed"]), 1)


if __name__ == "__main__":
    unittest.main()
