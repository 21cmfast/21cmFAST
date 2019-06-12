"""
A modification of the basic YAML and astropy.io.misc.yaml to be able to load/dump
objects with astropy quantities in them.
"""
import yaml
from astropy.io.misc import yaml as ayaml

class NewDumper(yaml.Dumper, ayaml.AstropyDumper):
    pass

class NewLoader(yaml.Loader, ayaml.AstropyLoader):
    pass


for k, v in yaml.Dumper.yaml_representers.items():
    NewDumper.add_representer(k, v)

for k, v in yaml.Dumper.yaml_multi_representers.items():
    NewDumper.add_multi_representer(k, v)

for k, v in ayaml.AstropyDumper.yaml_representers.items():
    NewDumper.add_representer(k, v)

for k, v in ayaml.AstropyDumper.yaml_multi_representers.items():
    NewDumper.add_multi_representer(k, v)

for k, v in yaml.Loader.yaml_constructors.items():
    NewLoader.add_constructor(k, v)

for k, v in ayaml.AstropyLoader.yaml_constructors.items():
    NewLoader.add_constructor(k, v)

for k, v in yaml.Loader.yaml_multi_constructors.items():
    NewLoader.add_multi_constructor(k, v)

for k, v in ayaml.AstropyLoader.yaml_multi_constructors.items():
    NewLoader.add_multi_constructor(k, v)

def load(stream):
    return yaml.load(stream, Loader=NewLoader)


def dump(data, stream=None, **kwargs):
    kwargs['Dumper'] = NewDumper
    return yaml.dump(data, stream=stream, **kwargs)