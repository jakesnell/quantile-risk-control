# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="var_control",
    version="0.0.1",
    description="Distribution-free guarantees for loss @ quantiles",
    long_description=readme,
    author="Jake Snell and Tom Zollo",
    author_email="js2523@princeton.edu, tpz2105@columbia.edu",
    url="https://github.com/jakesnell/quantile-risk-control",
    license=license,
    packages=find_packages(exclude=("tests")),
)
