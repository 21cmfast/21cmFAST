"""Modified YAML that can load/dump `astropy` quantities.

A modification of the basic YAML and astropy.io.misc.yaml to be able to load/dump
objects with astropy quantities in them.
"""

from typing import IO, Any, overload

import yaml
from astropy.io.misc import yaml as ayaml


class _NewDumper(yaml.Dumper, ayaml.AstropyDumper):
    pass


class _NewLoader(yaml.Loader, ayaml.AstropyLoader):
    pass


for k, v in yaml.Dumper.yaml_representers.items():
    _NewDumper.add_representer(k, v)

for k, v in yaml.Dumper.yaml_multi_representers.items():
    _NewDumper.add_multi_representer(k, v)

for k, v in ayaml.AstropyDumper.yaml_representers.items():
    _NewDumper.add_representer(k, v)

for k, v in ayaml.AstropyDumper.yaml_multi_representers.items():
    _NewDumper.add_multi_representer(k, v)

for k, v in yaml.Loader.yaml_constructors.items():
    _NewLoader.add_constructor(k, v)

for k, v in ayaml.AstropyLoader.yaml_constructors.items():
    _NewLoader.add_constructor(k, v)

for k, v in yaml.Loader.yaml_multi_constructors.items():
    _NewLoader.add_multi_constructor(k, v)

for k, v in ayaml.AstropyLoader.yaml_multi_constructors.items():
    _NewLoader.add_multi_constructor(k, v)


def load(stream: str | bytes | IO[str] | IO[bytes]) -> Any:
    """Load an object from a YAML stream.

    The returned object is whatever the YAML document describes (it is Any since it
    is decoded from the stream).
    """
    return yaml.load(stream, Loader=_NewLoader)


@overload
def dump(data: object, stream: None = None, **kwargs: Any) -> str | bytes: ...
@overload
def dump(data: object, stream: IO[str] | IO[bytes], **kwargs: Any) -> None: ...
def dump(
    data: object, stream: IO[str] | IO[bytes] | None = None, **kwargs: Any
) -> str | bytes | None:
    """Dump an object into a YAML stream.

    If no stream is given, the YAML document is returned (as bytes if an
    ``encoding`` is given, otherwise as a str). Other keyword arguments are passed
    to :func:`yaml.dump`.
    """
    kwargs["Dumper"] = _NewDumper
    return yaml.dump(data, stream=stream, **kwargs)
