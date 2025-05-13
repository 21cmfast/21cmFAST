#!/bin/python
"""
install_custom.py provides a custom installation process for the 21cmFAST package.

Allows users to specify various build and configuration options via command-line arguments.

Features:
- Allows setting the log level for the installation process.
- Provides an option to enable debug symbols for the build.
- Enables customization of the optimization level for the build process.

Command-line Arguments:
- --log-level: Specifies the log level for the build process. Options include:
    NO_LOG, ERROR, WARNING, INFO, DEBUG, SUPER_DEBUG, ULTRA_DEBUG. Defaults to WARNING.
- --debug: Enables debug symbols for the build, which can be useful for debugging.
- -o, --optimization: Sets the optimization level for the build (e.g., -O0, -O1, -O2, -O3).
    Defaults to 3.

Usage:
Run the script from the command line to install 21cmFAST with the desired options:
        python install_custom.py [options]

Example:
        python install_custom.py --log-level DEBUG --debug -o 2

Dependencies:
- Python 3.x
- pip (Python package installer)

Note:
This script uses the `subprocess` module to invoke the `pip install` command with
custom configuration settings.

"""

import argparse
import subprocess

# Define the command-line arguments
parser = argparse.ArgumentParser(description="Install 21cmFAST with custom options.")
parser.add_argument(
    "--log-level",
    type=str,
    default="WARNING",
    help="Set the log level (NO_LOG, ERROR, WARNING, INFO, DEBUG, SUPER_DEBUG, ULTRA_DEBUG)",
)
parser.add_argument("--debug", action="store_true", help="Enable debug symbols")
parser.add_argument(
    "-o",
    "--optimization",
    help="optimisation level (i,e -O0, -O1, -O2, -O3)",
    default="3",
)

args = parser.parse_args()

# Get the LOG_LEVEL environment variable (default to 'WARNING' if not set)
log_level_str = args.log_level
setup_args = [
    f"--config-setting=setup-args=-Dlog_level={log_level_str}",
]

if args.debug:
    setup_args += ["--config-setting=setup-args=-Dbuildtype=debugoptimized"]  # -O2

setup_args += [f"--config-setting=setup-args=-Doptimization={args.optimization}"]


# Run pip install with the specified options
subprocess.run(["pip", "install", ".", *setup_args])
