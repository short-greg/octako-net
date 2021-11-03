from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='octako',
    version='0.1.0',
    packages=['octako', 'octako.extensions', 'octako.machinery', 'octako.modules', 'octako.teaching']
)
