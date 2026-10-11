"""Settings: frozen dataclasses with no defaults, checked in both directions.

Rules (plan sections 6.1 and 6.3):
- Every setting a component uses is a field of its settings dataclass. There are no defaults,
  so a setting the notebook does not show cannot reach a model.
- A missing field raises. An unknown field raises. A value of the wrong type raises. That
  includes a number that arrives as text, such as "64# comment", the fault that stopped the
  summer run (plan 6.7), and a bool where a number is expected.
- An int is accepted where a float is declared and stored as a float, so 0 and 0.0 give the
  same hash. NaN and infinity are refused.
- Lists are stored as tuples, so validated settings cannot be changed afterwards.
- The validated settings are hashed from canonical JSON. The hash names the run.

Supported field types: bool, int, float, str, Literal[...] of str or int values,
list[X], tuple[X, ...], fixed tuple[X, Y], X | None, dict[str, X], and nested settings
dataclasses. Anything else is refused when the class is registered.

Fields added after runs exist (added 2026-09-26). A settings class may list, oldest first, the
fields it gained after runs were made with it, each with the value that does exactly what the
code did before the field existed:

    ADDED_FIELDS = (("2026-09-26", {"speed_factors": [1.0], "mixup_alpha": 0.0}),)

A settings dict must then hold either every field, or exactly the fields the class had before
one of those dates; a dict of the older form gets the listed values for the fields it lacks.
Nothing else is filled in, so a new sweep file that forgets one new field still fails. The hash
leaves out an added field while it holds its listed value (canonical_plain), so configurations
that do what older ones did keep the older hash, in either form, and runs stay comparable
across the change. A test runs both forms through the same code on the node.
"""
import dataclasses
import math
import re
import types
import typing

from affect.hashing import canonical_json, sha256_json


class SettingsError(ValueError):
    """A settings dict does not match its declaration."""


def _is_union(origin) -> bool:
    return origin is typing.Union or origin is types.UnionType


def _type_name(tp) -> str:
    if isinstance(tp, type):
        return tp.__name__
    return str(tp).replace("typing.", "")


def _check_type_supported(tp, where: str) -> None:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if tp in (bool, int, float, str):
        return
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        check_settings_class(tp)
        return
    if origin is typing.Literal:
        if not args:
            raise TypeError(f"{where}: empty Literal")
        for arg in args:
            if type(arg) not in (str, int):
                raise TypeError(f"{where}: Literal values must be str or int, got {arg!r}")
        return
    if _is_union(origin):
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != 1 or len(args) != 2:
            raise TypeError(f"{where}: only 'X | None' unions are supported, got {_type_name(tp)}")
        _check_type_supported(non_none[0], where)
        return
    if origin in (list, tuple):
        if origin is tuple and len(args) == 2 and args[1] is Ellipsis:
            _check_type_supported(args[0], where)
            return
        if origin is list and len(args) == 1:
            _check_type_supported(args[0], where)
            return
        if origin is tuple and args and Ellipsis not in args:
            for arg in args:
                _check_type_supported(arg, where)
            return
    if origin is dict and len(args) == 2 and args[0] is str:
        _check_type_supported(args[1], where)
        return
    raise TypeError(f"{where}: unsupported settings type {_type_name(tp)}")


def check_settings_class(cls) -> None:
    """Raise TypeError unless cls is a frozen dataclass with no defaults and supported types."""
    if not (isinstance(cls, type) and dataclasses.is_dataclass(cls)):
        raise TypeError(f"{cls!r} is not a dataclass")
    if not cls.__dataclass_params__.frozen:
        raise TypeError(f"{cls.__name__} must be declared with @dataclass(frozen=True)")
    hints = typing.get_type_hints(cls)
    for field in dataclasses.fields(cls):
        if field.default is not dataclasses.MISSING or field.default_factory is not dataclasses.MISSING:
            raise TypeError(f"{cls.__name__}.{field.name} has a default. Settings have no defaults.")
        _check_type_supported(hints[field.name], f"{cls.__name__}.{field.name}")
    added_generations(cls)


def added_generations(cls) -> tuple:
    """The class's ADDED_FIELDS, checked: ((date, {field: value before the field existed}), ...),
    dates as YYYY-MM-DD in increasing order, each field declared, listed once, not a nested
    settings class, and its value valid for the field's type. () when the class lists none."""
    generations = cls.__dict__.get("ADDED_FIELDS", ())
    if type(generations) is not tuple:
        raise TypeError(f"{cls.__name__}.ADDED_FIELDS must be a tuple of (date, {{field: value}}) pairs")
    names = {f.name for f in dataclasses.fields(cls)}
    hints = typing.get_type_hints(cls)
    seen, last = set(), ""
    for item in generations:
        if type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or type(item[1]) is not dict:
            raise TypeError(f"{cls.__name__}.ADDED_FIELDS: {item!r} is not a (date, {{field: value}}) pair")
        date, fields = item
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or date <= last:
            raise TypeError(f"{cls.__name__}.ADDED_FIELDS: dates must be YYYY-MM-DD, oldest first, got {date!r}")
        if not fields:
            raise TypeError(f"{cls.__name__}.ADDED_FIELDS: {date} lists no fields")
        for name, value in fields.items():
            if name not in names:
                raise TypeError(f"{cls.__name__}.ADDED_FIELDS: {name!r} is not a field")
            if name in seen:
                raise TypeError(f"{cls.__name__}.ADDED_FIELDS: {name!r} is listed twice")
            if isinstance(hints[name], type) and dataclasses.is_dataclass(hints[name]):
                raise TypeError(f"{cls.__name__}.ADDED_FIELDS: {name!r} is a nested settings class")
            try:
                _coerce(hints[name], value, f"{cls.__name__}.{name}")
            except SettingsError as err:
                raise TypeError(f"{cls.__name__}.ADDED_FIELDS: the value before {date} is not valid: {err}") from None
            seen.add(name)
        last = date
    return generations


def _forms(cls) -> list:
    """[(field names, {field: value filled in}), ...]: the current form first, then the form
    before each ADDED_FIELDS date, newest first."""
    names = [f.name for f in dataclasses.fields(cls)]
    forms = [(frozenset(names), {})]
    missing = {}
    for date, fields in reversed(added_generations(cls)):
        missing = {**missing, **fields}
        forms.append((frozenset(n for n in names if n not in missing), dict(missing)))
    return forms


def _values_before(cls) -> dict:
    """{added field: its value before it existed, validated and as plain JSON types}."""
    hints = typing.get_type_hints(cls)
    out = {}
    for _, fields in added_generations(cls):
        for name, value in fields.items():
            out[name] = to_plain(_coerce(hints[name], value, f"{cls.__name__}.{name}"))
    return out


def _coerce(tp, value, where: str):
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if isinstance(tp, type) and dataclasses.is_dataclass(tp):
        return validate(tp, value, where)
    if origin is typing.Literal:
        for arg in args:
            if type(value) is type(arg) and value == arg:
                return value
        raise SettingsError(f"{where}: {value!r} is not one of {list(args)}")
    if _is_union(origin):
        if value is None:
            return None
        inner = [a for a in args if a is not type(None)][0]
        return _coerce(inner, value, where)
    if tp is bool:
        if type(value) is not bool:
            raise SettingsError(f"{where}: expected true or false, got {type(value).__name__} {value!r}")
        return value
    if tp is int:
        if type(value) is not int:
            raise SettingsError(f"{where}: expected a whole number, got {type(value).__name__} {value!r}")
        return value
    if tp is float:
        if type(value) not in (int, float):
            raise SettingsError(f"{where}: expected a number, got {type(value).__name__} {value!r}")
        number = float(value)
        if not math.isfinite(number):
            raise SettingsError(f"{where}: {value!r} is not a finite number")
        return number
    if tp is str:
        if type(value) is not str:
            raise SettingsError(f"{where}: expected text, got {type(value).__name__} {value!r}")
        return value
    if origin in (list, tuple):
        if type(value) not in (list, tuple):
            raise SettingsError(f"{where}: expected a list, got {type(value).__name__} {value!r}")
        if origin is tuple and not (len(args) == 2 and args[1] is Ellipsis):
            if len(value) != len(args):
                raise SettingsError(f"{where}: expected {len(args)} values, got {len(value)}")
            return tuple(_coerce(a, v, f"{where}[{i}]") for i, (a, v) in enumerate(zip(args, value)))
        item_type = args[0]
        return tuple(_coerce(item_type, v, f"{where}[{i}]") for i, v in enumerate(value))
    if origin is dict:
        if type(value) is not dict:
            raise SettingsError(f"{where}: expected a mapping, got {type(value).__name__} {value!r}")
        out = {}
        for key, item in value.items():
            if type(key) is not str:
                raise SettingsError(f"{where}: keys must be text, got {key!r}")
            out[key] = _coerce(args[1], item, f"{where}.{key}")
        return out
    raise TypeError(f"{where}: unsupported settings type {_type_name(tp)}")


def validate(cls, raw, where: str):
    """Check a settings dict against its dataclass and return the validated instance.

    `where` names the location for error messages, for example "model".
    """
    check_settings_class(cls)
    if type(raw) is not dict:
        raise SettingsError(f"{where}: expected a mapping of settings, got {type(raw).__name__}")
    declared = [f.name for f in dataclasses.fields(cls)]
    fill = next((values for names, values in _forms(cls) if frozenset(raw) == names), None)
    if fill is None:
        missing = [name for name in declared if name not in raw]
        unknown = sorted(key for key in raw if key not in declared)
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"unknown {unknown}")
        dates = [date for date, _ in added_generations(cls)]
        if dates:
            parts.append(f"(settings written before {', '.join(dates)} may instead hold exactly the fields of that time)")
        raise SettingsError(f"{where} ({cls.__name__}): " + "; ".join(parts))
    raw = {**raw, **fill}
    hints = typing.get_type_hints(cls)
    values = {name: _coerce(hints[name], raw[name], f"{where}.{name}") for name in declared}
    return cls(**values)


def to_plain(obj):
    """Validated settings as plain JSON types: dataclasses become dicts, tuples become lists."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_plain(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    return obj


def canonical_plain(obj):
    """to_plain, leaving out every added field (ADDED_FIELDS) that holds its value from before it
    existed. This is what gets hashed: settings that do what older settings did hash as those did."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        before = _values_before(type(obj))
        out = {}
        for f in dataclasses.fields(obj):
            value = canonical_plain(getattr(obj, f.name))
            if f.name in before and canonical_json(value) == canonical_json(before[f.name]):
                continue
            out[f.name] = value
        return out
    if isinstance(obj, (list, tuple)):
        return [canonical_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: canonical_plain(v) for k, v in obj.items()}
    return obj


def settings_hash(obj) -> str:
    """SHA-256 of the canonical JSON of validated settings (canonical_plain)."""
    return sha256_json(canonical_plain(obj))


def describe(cls) -> list[tuple[str, str]]:
    """(field, type) pairs of a settings class, in declaration order, for display. An added field's
    type is followed by when it was added and its value before then."""
    check_settings_class(cls)
    hints = typing.get_type_hints(cls)
    added = {name: (date, value) for date, fields in added_generations(cls) for name, value in fields.items()}
    out = []
    for f in dataclasses.fields(cls):
        text = _type_name(hints[f.name])
        if f.name in added:
            date, value = added[f.name]
            text += f" (added {date}; before then {canonical_json(value)})"
        out.append((f.name, text))
    return out
