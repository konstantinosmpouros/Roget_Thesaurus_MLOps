import io
import os
from pathlib import Path

from setuptools import setup, find_packages

path = os.path.abspath(os.path.dirname(__file__))

# Metadata of package
NAME = 'roget-thesaurus-classification'
DESCRIPTION = "Pipelines that can predict the class of the section a word belong according to Roget's Thesaurus"
URL = 'https://github.com/konstantinosmpouros/Roget-Thesaurus-Classification-MLOps'
EMAIL = 'kostasbouros@hotmail.gr'
AUTHOR = 'Konstantinos Bouros'
REQUIRES_PYTHON = '>=3.10.12'


# Get the list of packages to be installed
def list_reqs(fname='requirements.txt'):
    with io.open(os.path.join(path, fname), encoding='utf-8') as f:
        return f.read().splitlines()


# Get the description in the README.md
try:
    with io.open(os.path.join(path, 'README.md'), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
ROOT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = ROOT_DIR / 'prediction_model'
about = {}
with open(PACKAGE_DIR / 'VERSION') as f:
    _version = f.read().strip()
    about['__version__'] = _version


setup(
    version=about['__version__'],
    name=NAME,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=URL,
    author_email=EMAIL,
    author=AUTHOR, 
    python_requires=REQUIRES_PYTHON,
    include_package_data=True,
    packages=find_packages(exclude=('tests',)),
    package_data={'prediction_model': ['VERSION']},
    install_requires=list_reqs(),
    extras_require={},
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python',
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        "Operating System :: POSIX :: Linux"
    ]
)