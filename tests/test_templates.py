"""Tests of the templates module."""
from py21cmfast import templates as tmpl
import pytest
from py21cmfast.wrapper.inputs import InputStruct
from pathlib import Path

_TEMPLATES = tmpl.list_templates()
_ALL_ALIASES = sum((t['alias'] for t in _TEMPLATES), [])

class TestListTemplates:
    def setup_class(self):
        self.templates = _TEMPLATES

    def test_output_form(self):
        assert isinstance(self.templates, list)
        assert all(isinstance(x, dict) for x in self.templates)

    def test_existence_of_files(self):
        assert all((tmpl.TEMPLATE_PATH / t['file']).exists() for t in self.templates)

    def test_uniqueness_of_names(self):
        allnames = [t['name'] for t in self.templates]
        assert len(set(allnames)) == len(allnames)

    def test_uniqueness_of_aliases(self):
        assert len(set(_ALL_ALIASES)) == len(_ALL_ALIASES)


class TestLoadTemplateFile:
    """Tests of the load_template_file function."""

    @pytest.mark.parametrize(
        'template',
        _ALL_ALIASES
    )
    def test_load_str(self, template: str):
        out = tmpl.load_template_file(template)
        possible = set(InputStruct._subclasses.keys())
        assert all(key in possible for key in out)

    def test_load_path(self, tmp_path: Path):
        pth = tmp_path / 'tmp.toml'
        with pth.open('w') as fl:
            fl.write(
                """
                [SimulationOptions]
                BOX_LEN = 100
                """
            )

        out = tmpl.load_template_file(pth)
        assert out['SimulationOptions']['BOX_LEN'] = 100


    def test_non_existent(self):
        with pytest.raises(ValueError, match='Template VIRIDISSUCKS not found on-disk'):
            tmpl.load_template_file("VIRIDISSUCKS")
