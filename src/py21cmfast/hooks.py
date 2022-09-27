"""Built-in hooks to be used when evolving boxes.

Hooks can be passed to high-level functions like :func:`~py21cmfast.wrapper.run_coeval`
and :func:`~py21cmfast.wrapper.run_lightcone`, or lower-level functions like
:func:`~py21cmfast.wrapper.spin_temp` using the ``hooks=`` parameter.

You may define your own hooks. They are simply python functions that take as their
first parameter an :class:`~py21cmfast.outputs._OutputStruct`, and any other parameters
they wish after that. They should not return anything, but may write out files or
attach data to the output box.
"""

import numpy as np
from pathlib import Path
from typing import Union

from ._cfg import config
from .cache_tools import get_boxes_at_redshift
from .outputs import BrightnessTemp, _OutputStruct


def rolling_cache(
    box: _OutputStruct, cache_dir: Union[str, Path] = None, keep: tuple[str] = ()
):
    """A cache in which only required boxes for checkpoint restarts are kept."""
    cache_dir = Path(cache_dir or config["direc"])
    box.write(direc=cache_dir)

    if isinstance(box, BrightnessTemp):
        boxes = get_boxes_at_redshift(
            direc=cache_dir,
            redshift=(box.redshift, np.inf),
            user_params=box.user_params,
            cosmo_params=box.cosmo_params,
            astro_params=box.astro_params,
            flag_options=box.flag_options,
        )

        if not boxes:
            raise ValueError("Got not boxes!")
        for kind, bxs in boxes.items():
            if kind in keep:
                continue

            bxs = sorted(bxs, key=lambda x: x.redshift)
            if len(bxs) <= 2:
                if box.redshift < 33:
                    raise ValueError("Should have found more boxes.")
                continue

            for bx in bxs[2:]:
                (cache_dir / bx.filename).unlink()
