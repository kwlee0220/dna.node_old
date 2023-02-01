from __future__ import annotations

from typing import Any, Callable, List, Union
from dataclasses import dataclass, field


@dataclass(frozen=True, eq=True, repr=False)
class Option:
    value: Any
    president: bool = field(default=True)

    @staticmethod
    def empty() -> Option:
        return _EMPTY

    @staticmethod
    def of(value: Any) -> Option:
        return Option(value, True)

    @staticmethod
    def ofNullable(value: Any) -> Option:
        return Option(value, True) if value is not None else Option.empty()

    def get(self) -> Any:
        if self.president:
            return self.value
        else:
            raise ValueError("NoSuchValue")

    def getOrNone(self) -> Any:
        return self.value if self.president else None

    def getOrElse(self, else_value) -> Any:
        return self.value if self.president else else_value

    def getOrCall(self, else_value) -> Any:
        return self.value if self.president else else_value()

    def is_present(self) -> bool:
        return self.president

    def is_absent(self) -> bool:
        return self.president

    def if_present(self, call) -> Option:
        if self.president:
            call()

        return self

    def if_absent(self, call) -> Option:
        if not self.president:
            call()

        return self

    def map(self, mapper) -> Option:
        return Option.of(mapper(self.value)) if self.president else Option.empty()

    def transform(self, target:Any, mapper:Callable[[Any, Any], Any]) -> Option:
        if self.is_present:
            return mapper(target, self.value)
        else:
            return target

    def __repr__(self) -> str:
        return f'Option({self.value})' if self.president else 'Option(Empty)'

_EMPTY = Option(None, False)