from setuptools import setup, find_packages
import os

"""
This script serves as the `setup.py` file for the `afrl_challenge` Python package.

It uses `setuptools` to define the package metadata, dependencies, and build
instructions, allowing the project to be installed and distributed as a
standard Python package. This includes information such as the package name,
version, authors, a brief description, and a longer description read from
`README.md`.

The `setup()` function specifies:
- `name`: The name of the package.
- `version`: The current version of the package.
- `author`: The primary authors or contributors.
- `description`: A short summary of the package.
- `long_description`: A more detailed description, typically read from `README.md`.
- `long_description_content_type`: Specifies the format of the long description (e.g., Markdown).
- `packages`: Lists the Python packages that are part of the project.
- `classifiers`: Metadata that helps users find the project on PyPI.
- `python_requires`: The minimum Python version required.
- `install_requires`: A list of external Python packages that this project depends on.

To install this package in editable mode for development, run:
`pip install -e .`

To build a distributable package (e.g., wheel), run:
`python setup.py sdist bdist_wheel`
"""

def read_long_description():
    """
    Reads the content of the `README.md` file to be used as the long description
    for the package.

    If `README.md` is not found, it returns a default description string.
    """
    this_directory = os.path.abspath(os.path.dirname(__file__))
    try:
        with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "A Gymnasium environment for the AFRL 2026 Challenge Competition"

setup(
    name="afrl_challenge",
    version="2.0.0",
    author="corryn.collins, david.shoukr, alana.li, adis.delanovic, elizabeth.andreas, zane.kitchen-lipski",
    description="A Gymnasium environment for the AFRL 2026 Challenge Competition.",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "pygame>=2.5.0",
        "stable-baselines3[extra]",
        "torch",
        "protobuf==3.20.3",
        "seaborn>=0.13.2",
        "jupyter>=1.1.1",
        "imageio>=2.37.0"
    ],
)


