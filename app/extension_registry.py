"""Framework-neutral registries used by LASER extension points."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, Protocol, TypeVar


ComponentT = TypeVar("ComponentT")
OrderedT = TypeVar("OrderedT", bound="KeyedDefinition")


class KeyedDefinition(Protocol):
    """Small structural boundary shared by ordered UI/report definitions."""

    @property
    def key(self) -> str:
        """Stable lookup key for the ordered definition."""


def normalise_component_name(value: object) -> str:
    return str(value).strip().casefold().replace("_", "-")


@dataclass(frozen=True)
class ComponentRegistry(Generic[ComponentT]):
    """Immutable name-to-component registry with optional startup checks.

    ``expected_type`` may be a concrete class or a runtime-checkable Protocol.
    Runtime Protocol checks establish structural presence; static checking is
    still responsible for verifying precise method signatures.
    """

    _components: Mapping[str, ComponentT]
    _expected_type: type[object] | tuple[type[object], ...] | None
    _name_attribute: str | None

    def __init__(
        self,
        components: Mapping[str, ComponentT] | None = None,
        *,
        expected_type: type[object] | tuple[type[object], ...] | None = None,
        name_attribute: str | None = None,
    ):
        normalised: dict[str, ComponentT] = {}
        for name, component in (components or {}).items():
            key = normalise_component_name(name)
            if not key:
                raise ValueError("component name cannot be empty")
            if key in normalised:
                raise ValueError(
                    f"duplicate component name after normalization: {name!r}"
                )
            self._validate_component(
                key,
                component,
                expected_type=expected_type,
                name_attribute=name_attribute,
            )
            normalised[key] = component
        object.__setattr__(self, "_components", MappingProxyType(normalised))
        object.__setattr__(self, "_expected_type", expected_type)
        object.__setattr__(self, "_name_attribute", name_attribute)

    @staticmethod
    def _validate_component(
        key: str,
        component: object,
        *,
        expected_type: type[object] | tuple[type[object], ...] | None,
        name_attribute: str | None,
    ) -> None:
        if expected_type is not None and not isinstance(component, expected_type):
            expected = (
                ", ".join(item.__name__ for item in expected_type)
                if isinstance(expected_type, tuple)
                else expected_type.__name__
            )
            raise TypeError(
                f"component {key!r} does not satisfy expected type {expected}"
            )
        if name_attribute is None:
            return
        component_name = getattr(component, name_attribute, None)
        if normalise_component_name(component_name) != key:
            raise ValueError(
                f"component {key!r} must expose matching {name_attribute} "
                f"{key!r}; received {component_name!r}"
            )

    def with_component(
        self, name: str, component: ComponentT,
    ) -> "ComponentRegistry[ComponentT]":
        key = normalise_component_name(name)
        if not key:
            raise ValueError("component name cannot be empty")
        if key in self._components:
            raise ValueError(f"component already registered: {name}")
        components = dict(self._components)
        components[key] = component
        return ComponentRegistry(
            components,
            expected_type=self._expected_type,
            name_attribute=self._name_attribute,
        )

    def replace_component(
        self, name: str, component: ComponentT,
    ) -> "ComponentRegistry[ComponentT]":
        """Return a registry with an explicitly replaced existing component."""
        key = normalise_component_name(name)
        if key not in self._components:
            raise ValueError(f"cannot replace unregistered component: {name}")
        components = dict(self._components)
        components[key] = component
        return ComponentRegistry(
            components,
            expected_type=self._expected_type,
            name_attribute=self._name_attribute,
        )

    def require(self, name: object) -> ComponentT:
        key = normalise_component_name(name)
        try:
            return self._components[key]
        except KeyError as exc:
            available = ", ".join(sorted(self._components)) or "none"
            raise ValueError(
                f"Unknown component {name!r}; available components: {available}."
            ) from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._components))

    def items(self) -> Iterable[tuple[str, ComponentT]]:
        return self._components.items()

    def __iter__(self) -> Iterator[str]:
        return iter(self._components)

    def __contains__(self, name: object) -> bool:
        return normalise_component_name(name) in self._components


@dataclass(frozen=True)
class OrderedRegistry(Generic[OrderedT]):
    """Immutable ordered registry for definitions carrying a stable ``key``."""

    items: tuple[OrderedT, ...]

    def __init__(self, items: Iterable[OrderedT] = ()):
        ordered = tuple(items)
        keys = [normalise_component_name(item.key) for item in ordered]
        if any(not key for key in keys):
            raise ValueError("ordered item key cannot be empty")
        if len(keys) != len(set(keys)):
            raise ValueError("ordered item keys must be unique")
        object.__setattr__(self, "items", ordered)

    def with_item(
        self,
        item: OrderedT,
        *,
        after: str | None = None,
    ) -> "OrderedRegistry[OrderedT]":
        key = normalise_component_name(item.key)
        if key in self:
            raise ValueError(f"ordered item already registered: {item.key}")
        items = list(self.items)
        if after is None:
            items.append(item)
        else:
            target = normalise_component_name(after)
            try:
                index = next(
                    index
                    for index, existing in enumerate(items)
                    if normalise_component_name(existing.key) == target
                )
            except StopIteration as exc:
                raise ValueError(f"unknown ordered item: {after}") from exc
            items.insert(index + 1, item)
        return OrderedRegistry(items)

    def __iter__(self) -> Iterator[OrderedT]:
        return iter(self.items)

    def __contains__(self, key: object) -> bool:
        normalized = normalise_component_name(key)
        return any(
            normalise_component_name(item.key) == normalized
            for item in self.items
        )


__all__ = [
    "ComponentRegistry",
    "KeyedDefinition",
    "OrderedRegistry",
    "normalise_component_name",
]
