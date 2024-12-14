"""Classes dealing with the state of arrays."""

import attrs


class ArrayStateError(ValueError):
    """Errors arising from incorrectly modifying array state."""

    pass


@attrs.define(frozen=True)
class ArrayState:
    """Define the memory state of a struct array."""

    initialized = attrs.field(default=False, converter=bool)
    c_memory = attrs.field(default=False, converter=bool)
    computed_in_mem = attrs.field(default=False, converter=bool)
    on_disk = attrs.field(default=False, converter=bool)

    def deinitialize(self):
        return attrs.evolve(initialized=False, computed_in_mem=False)

    def initialize(self):
        return attrs.evolve(self, initialized=True)

    def as_computed(self):
        return attrs.evolve(self, computed_in_mem=True, initialized=True)

    def dropped(self):
        return attrs.evolve(self, initialized=False, computed_in_mem=False)

    def written(self):
        return attrs.evolve(self, on_disk=True)

    def purged_to_disk(self):
        return attrs.evolve(
            self, initialized=False, computed_in_mem=False, on_disk=True
        )

    def loaded_from_disk(self):
        return attrs.evolve(self, initialized=True, computed_in_mem=True, on_disk=True)

    @property
    def computed(self):
        """Whether the array is computed anywhere."""
        return self.computed_in_mem or self.on_disk

    @property
    def c_has_active_memory(self):
        """Whether C currently has initialized memory for this array."""
        return self.c_memory and self.initialized

    def __str__(self):
        """Returns a string representation of the ArrayState."""
        if self.computed_in_mem:
            return "computed (in mem)"
        elif self.on_disk:
            return "computed (on disk)"
        elif self.initialized:
            return "memory initialized (not computed)"
        else:
            return "uncomputed and uninitialized"
