"""Shared helpers for the tests."""
import importlib.util
import os
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def required() -> set:
    return {x.strip() for x in os.environ.get("AFFECT_REQUIRE", "").split(",") if x.strip()}


def _skip_or_fail(reason: str, must_run: bool):
    """Skip, or, when this machine must run the test (AFFECT_REQUIRE), fail every test it covers.
    Works on test methods and on whole test classes."""
    if not must_run:
        return unittest.skip(reason)
    message = f"{reason}, but AFFECT_REQUIRE says this machine must run it"

    def decorator(obj):
        if isinstance(obj, type):
            obj.setUpClass = classmethod(lambda cls: None)

            def setUp(self):
                self.fail(message)

            obj.setUp = setUp
            return obj

        def failing(self, *args, **kwargs):
            self.fail(message)

        failing.__name__ = obj.__name__
        return failing

    return decorator


def needs_module(module: str):
    """Skip when a module is missing, unless AFFECT_REQUIRE names it; then fail."""
    if importlib.util.find_spec(module) is not None:
        return lambda obj: obj
    return _skip_or_fail(f"{module} is not installed", module in required())


def needs_assembled(dataset: str):
    """Skip when data/<dataset>/assembled/ is not here, unless AFFECT_REQUIRE has 'data'."""
    if (REPO / "data" / dataset / "assembled" / "ASSEMBLY.json").is_file():
        return lambda obj: obj
    return _skip_or_fail(f"data/{dataset}/assembled is not present", "data" in required())
