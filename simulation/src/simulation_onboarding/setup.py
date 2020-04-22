# ! DO NOT MANUALLY INVOKE THIS setup.py, USE CATKIN INSTEAD

# - Beginning sphinx -
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
# fmt: off
setup_args = generate_distutils_setup(
    # Packages contains all foldernames inside the package_dir
    packages=(
        [
            'onboarding',  # Include package with onboarding node
        ]
    ),
    package_dir={"": "src"},
)
# fmt: on

setup(**setup_args)
