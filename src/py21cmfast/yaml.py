"""Modified YAML that can load/dump `astropy` quantities.

A modification of the basic YAML and astropy.io.misc.yaml to be able to load/dump
objects with astropy quantities in them.
"""
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


def load(stream):
    """Load an object from a YAML stream."""
    return yaml.load(stream, Loader=_NewLoader)


def dump(data, stream=None, **kwargs):
    """Dump an object into a YAML stream."""
    kwargs["Dumper"] = _NewDumper
    return yaml.dump(data, stream=stream, **kwargs)
