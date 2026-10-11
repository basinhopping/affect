"""Layout rules from plan 6.1 and 6.7, checked on the tree itself.

- No .py file outside src/ and tests/: one-off scripts at the root are how the summer project
  grew 59 root scripts that copied logic and wrote shared files.
- No module-level numeric constants in the package: settings live only in validated settings
  objects (the summer project defined SEED in 21 scripts and IG_STEPS in 15).
- Every *Settings class in the package is a frozen dataclass with no defaults.
"""
import ast
import importlib
import inspect
import os
import pkgutil
import unittest
from pathlib import Path

from tests.support import REPO

# Folders that hold data, results or environments rather than code.
NOT_CODE = {"data", "artifacts", "queue", "logs", ".git", ".venv", "venv", "__pycache__", ".ipynb_checkpoints"}


class TestLayout(unittest.TestCase):
    def test_no_python_files_outside_src_and_tests(self):
        offenders = []
        for folder, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs if d not in NOT_CODE]   # never walk the data
            rel_folder = Path(folder).relative_to(REPO)
            top = rel_folder.parts[0] if rel_folder.parts else ""
            for name in files:
                if name.endswith(".py") and top not in ("src", "tests"):
                    offenders.append((rel_folder / name).as_posix())
        self.assertEqual(offenders, [], "Python files belong in src/affect or tests/")

    def test_no_module_level_numeric_constants(self):
        offenders = []
        for path in (REPO / "src" / "affect").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:
                targets, value = [], None
                if isinstance(node, ast.Assign):
                    targets, value = node.targets, node.value
                elif isinstance(node, ast.AnnAssign) and node.value is not None:
                    targets, value = [node.target], node.value
                if value is None:
                    continue
                if isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.USub):
                    value = value.operand
                if isinstance(value, ast.Constant) and type(value.value) in (int, float):
                    names = [t.id for t in targets if isinstance(t, ast.Name)]
                    offenders.append(f"{path.relative_to(REPO).as_posix()}: {names} = {value.value}")
        self.assertEqual(offenders, [], "numbers that act as settings belong in a settings dataclass")

    def test_settings_classes_are_frozen_without_defaults(self):
        import affect
        from affect.settings import check_settings_class

        checked = 0
        for info in pkgutil.walk_packages(affect.__path__, "affect."):
            module = importlib.import_module(info.name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if name.endswith("Settings") and obj.__module__ == module.__name__:
                    check_settings_class(obj)
                    checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()
