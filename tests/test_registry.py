import dataclasses
import unittest

from affect.registry import Registry
from affect.settings import SettingsError


@dataclasses.dataclass(frozen=True)
class S:
    width: int


class TestRegistry(unittest.TestCase):
    def test_register_get_and_validate_section(self):
        reg = Registry("model")

        @reg.register("tiny", settings=S)
        def build(settings):
            return settings.width

        entry, settings = reg.validate_section({"name": "tiny", "width": 8}, "model")
        self.assertEqual(entry.target(settings), 8)

    def test_duplicate_name_raises(self):
        reg = Registry("model")
        reg.register("tiny", settings=S)(lambda s: s)
        with self.assertRaises(ValueError):
            reg.register("tiny", settings=S)(lambda s: s)

    def test_unknown_name_lists_registered(self):
        reg = Registry("model")
        reg.register("tiny", settings=S)(lambda s: s)
        with self.assertRaisesRegex(SettingsError, "registered: \\['tiny'\\]"):
            reg.validate_section({"name": "huge", "width": 8}, "model")

    def test_bad_settings_class_fails_at_registration(self):
        @dataclasses.dataclass(frozen=True)
        class WithDefault:
            width: int = 3

        reg = Registry("model")
        with self.assertRaises(TypeError):
            reg.register("tiny", settings=WithDefault)

    def test_iemocap_is_registered(self):
        from affect import catalog
        from affect.registry import DATASETS

        catalog.load_all()
        self.assertIn("iemocap", DATASETS.names())


if __name__ == "__main__":
    unittest.main()
