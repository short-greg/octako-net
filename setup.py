from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='takonet',
    version='0.1.0',
    packages=['takonet', 'takonet.extensions', 'takonet.machinery', 'takonet.modules', 'takonet.teaching']
)
