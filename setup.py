"""Packaging settings."""

from codecs import open
from os.path import abspath, dirname, join
from subprocess import call

from setuptools import Command, find_packages, setup

from nntool import __version__


this_dir = abspath(dirname(__file__))
with open(join(this_dir, 'README.md'), encoding='utf-8') as file:
    long_description = file.read()


long_description = """
    Training and view perfomance of Deep Neural Networks with .txt topology file
    and little piece of code.
"""
setup(
    name='nntool',
    version=__version__,
    description='Deep Learning analyzer',
    long_description=long_description,
    url='https://github.com/EfMichalis/nntool',
    author='Efthymis Michalis',
    author_email='mefthymis@gmail.com',
    license='UNLICENSE',
    classifiers=[
        'Development Status :: 3 - Alpha'
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: Public Domain',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization'
    ],
    # keywords = 'cli',
    packages=['nntool'],
    package_dir={'nntool': 'nntool'},
    install_requires=[
        'docopt',
        'tensorflow',
        'scipy==1.1.0',
        'matplotlib',
        'pandas',
        'scikit-image',
        'numpy'
    ],
    entry_points={
        'console_scripts': [
            'nntool = nntool.cli:main'
        ],
    },
    test_suite='nose.collector',
    tests_require=['nose'],
    # extras_require = {
    #     'test': ['coverage', 'pytest', 'pytest-cov'],
    # },
)
