"""Classes dealing with the state of arrays."""


class ArrayStateError(ValueError):
    """Errors arising from incorrectly modifying array state."""

    pass


class ArrayState:
    """Define the memory state of a struct array."""

    def __init__(
        self, initialized=False, c_memory=False, computed_in_mem=False, on_disk=False
    ):
        self._initialized = initialized
        self._c_memory = c_memory
        self._computed_in_mem = computed_in_mem
        self._on_disk = on_disk

    @property
    def initialized(self):
        """Whether the array is initialized (i.e. allocated memory)."""
        return self._initialized

    @initialized.setter
    def initialized(self, val):
        if not val:
            # if its not initialized, can't be computed in memory
            self.computed_in_mem = False
        self._initialized = bool(val)

    @property
    def c_memory(self):
        """Whether the array's memory (if any) is controlled by C."""
        return self._c_memory

    @c_memory.setter
    def c_memory(self, val):
        self._c_memory = bool(val)

    @property
    def computed_in_mem(self):
        """Whether the array is computed and stored in memory."""
        return self._computed_in_mem

    @computed_in_mem.setter
    def computed_in_mem(self, val):
        if val:
            # any time we pull something into memory, it must be initialized.
            self.initialized = True
        self._computed_in_mem = bool(val)

    @property
    def on_disk(self):
        """Whether the array is computed and store on disk."""
        return self._on_disk

    @on_disk.setter
    def on_disk(self, val):
        self._on_disk = bool(val)

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
