# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
# fmt: off
setup_args = generate_distutils_setup(
    packages=(
        [
            "speaker",
            "evaluation_test",
            "state_machine",
            "referee",
        ]
    ),
    package_dir={"": "src"},
)
# fmt: on

setup(**setup_args)
