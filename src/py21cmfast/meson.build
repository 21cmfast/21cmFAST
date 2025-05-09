source_files = [
  '__init__.py',
  '_cfg.py',
  '_logging.py',
  'cli.py',
  'lightcones.py',
  'plotting.py',
  'run_templates.py',
  'utils.py',
  'yaml.py',
]

py.install_sources(
    source_files,
    subdir: 'py21cmfast'
)

pure_subdirs = [
  'drivers',
  'io',
  'wrapper',
  '_data',
]

# Install the Python code
install_root = py.get_install_dir()
foreach subdir: pure_subdirs
  install_subdir(subdir, install_dir: install_root / 'py21cmfast')
endforeach

# Build C-extension
subdir('src')
