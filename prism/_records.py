"""Small record classes without source-level type annotations."""

import copy


class _Missing:
    pass


MISSING = _Missing()


class RecordField:

    def __init__(self, default = MISSING):
        self.default = default


def field(default = MISSING):
    return RecordField(default)


def record(cls):
    fields = []
    for base in reversed(cls.__mro__):
        for name, value in vars(base).items():
            if isinstance(value, RecordField):
                fields.append((name, value.default))

    names = [name for name, _ in fields]

    def initialize(self, *args, **kwargs):
        if len(args) > len(fields):
            raise TypeError(
                "%s accepts at most %d positional arguments" %
                (cls.__name__, len(fields))
            )
        values = dict(zip(names, args))
        unknown = set(kwargs).difference(names)
        if unknown:
            raise TypeError(
                "unexpected argument(s) for %s: %s" %
                (cls.__name__, ", ".join(sorted(unknown)))
            )
        duplicate = set(values).intersection(kwargs)
        if duplicate:
            raise TypeError(
                "multiple values for %s" % ", ".join(sorted(duplicate))
            )
        values.update(kwargs)
        missing = []
        for name, default in fields:
            if name in values:
                value = values[name]
            elif default is MISSING:
                missing.append(name)
                continue
            else:
                value = copy.deepcopy(default)
            setattr(self, name, value)
        if missing:
            raise TypeError(
                "missing required argument(s) for %s: %s" %
                (cls.__name__, ", ".join(missing))
            )

    def represent(self):
        values = ", ".join(
            "%s=%r" % (name, getattr(self, name)) for name, _ in fields
        )
        return "%s(%s)" % (cls.__name__, values)

    def equal(self, other):
        return type(self) is type(other) and all(
            getattr(self, name) == getattr(other, name) for name, _ in fields
        )

    cls.__init__ = initialize
    cls.__repr__ = represent
    cls.__eq__ = equal
    cls.__record_fields__ = tuple(names)
    return cls


def asdict(value):
    if not hasattr(value, "__record_fields__"):
        raise TypeError("asdict expects a record instance")
    return {
        name: copy.deepcopy(getattr(value, name))
        for name in value.__record_fields__
    }


def replace(value, **changes):
    values = asdict(value)
    unknown = set(changes).difference(values)
    if unknown:
        raise TypeError("unknown record field(s): %s" % ", ".join(sorted(unknown)))
    values.update(changes)
    return type(value)(**values)
