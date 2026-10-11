"""Tests. Run from the repository root with the standard library's runner:

    PYTHONPATH=src python -m unittest discover -s tests -t .

Tests that need something a machine may lack (torch, scipy, the assembled data) skip there.
Set AFFECT_REQUIRE to a comma-separated list (torch,scipy,data) to turn those skips into
failures; scripts/check.sh does that on the node, so nothing is silently skipped there.
"""
