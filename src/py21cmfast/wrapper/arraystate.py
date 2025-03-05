"""Classes dealing with the state of arrays."""

from typing import Self

import attrs


class ArrayStateError(ValueError):
    """Errors arising from incorrectly modifying array state."""


@attrs.define(frozen=True)
class ArrayState:
    """Define the memory state of a struct array."""

    initialized = attrs.field(default=False, converter=bool)
    c_memory = attrs.field(default=False, converter=bool)
    computed_in_mem = attrs.field(default=False, converter=bool)
    on_disk = attrs.field(default=False, converter=bool)

    def deinitialize(self) -> Self:
        """Return new state that is not initialized."""
        return attrs.evolve(self, initialized=False, computed_in_mem=False)

    def initialize(self) -> Self:
        """Return new state that is initialized."""
        return attrs.evolve(self, initialized=True)

    def computed(self) -> Self:
        """Return new state indicating the array has been computed."""
        return attrs.evolve(self, computed_in_mem=True, initialized=True)

    def dropped(self) -> Self:
        """Return new state indicating the array has been dropped from memory."""
        return attrs.evolve(self, initialized=False, computed_in_mem=False)

    def written(self) -> Self:
        """Return new state indicating the array has been written to disk."""
        return attrs.evolve(self, on_disk=True)

    def purged_to_disk(self) -> Self:
        """Return new state indicating the array has been written to disk and dropped."""
        return self.written().dropped()

    def loaded_from_disk(self) -> Self:
        """Return new state indicating the array has been loaded from disk into memory."""
        return self.computed().written()

    @property
    def is_computed(self) -> bool:
        """Whether the array is computed anywhere."""
        return self.computed_in_mem or self.on_disk

    @property
    def c_has_active_memory(self) -> bool:
        """Whether C currently has initialized memory for this array."""
        return self.c_memory and self.initialized

    def __str__(self) -> str:
        """Return a string representation of the ArrayState."""
        if self.computed_in_mem:
            return "computed (in mem)"
        elif self.on_disk:
            return "computed (on disk)"
        elif self.initialized:
            return "memory initialized (not computed)"
        else:
            return "uncomputed and uninitialized"
