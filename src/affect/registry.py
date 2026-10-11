"""Registries: a dataset, feature or model is a module plus one registry entry.

A component registers itself with a decorator that names it and declares its settings class:

    @register_model("cnn", settings=CNNSettings)
    def build_cnn(settings, n_classes): ...

The settings class is checked when the module is imported (frozen dataclass, no defaults,
supported types), so a bad declaration fails at import rather than in the middle of a run.
Components are imported by affect.catalog.load_all(), which lists every module once.
"""
import dataclasses

from affect.settings import SettingsError, check_settings_class, validate


@dataclasses.dataclass(frozen=True)
class Entry:
    kind: str
    name: str
    settings_class: type
    target: object


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._entries: dict[str, Entry] = {}

    def register(self, name: str, settings: type):
        if not name or not name.replace("_", "").isalnum() or name != name.lower():
            raise ValueError(f"{self.kind} name {name!r}: use lowercase letters, digits and _")
        check_settings_class(settings)

        def decorator(target):
            if name in self._entries:
                raise ValueError(f"{self.kind} {name!r} is registered twice")
            self._entries[name] = Entry(self.kind, name, settings, target)
            return target

        return decorator

    def get(self, name: str) -> Entry:
        if name not in self._entries:
            raise KeyError(f"no {self.kind} named {name!r}; registered: {sorted(self._entries)}")
        return self._entries[name]

    def names(self) -> list[str]:
        return sorted(self._entries)

    def validate_section(self, raw, where: str):
        """Validate a settings section of the form {"name": <registered name>, **fields}.

        Returns (entry, validated settings). The name selects the settings class; every other
        key must be a field of that class.
        """
        if type(raw) is not dict:
            raise SettingsError(f"{where}: expected a mapping, got {type(raw).__name__}")
        if "name" not in raw:
            raise SettingsError(f"{where}: missing 'name' (one of {self.names()})")
        name = raw["name"]
        if type(name) is not str:
            raise SettingsError(f"{where}.name: expected text, got {name!r}")
        try:
            entry = self.get(name)
        except KeyError as err:
            raise SettingsError(f"{where}.name: {err.args[0]}") from None
        fields = {k: v for k, v in raw.items() if k != "name"}
        return entry, validate(entry.settings_class, fields, where)


DATASETS = Registry("dataset")
FEATURES = Registry("feature")
MODELS = Registry("model")
CLASSICAL = Registry("classical method")      # added 2026-09-26: affect.classical (SVM, TPOT)

register_dataset = DATASETS.register
register_feature = FEATURES.register
register_model = MODELS.register
register_classical = CLASSICAL.register
