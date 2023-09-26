from setuptools import setup, find_packages

with open('requirements.txt', encoding='utf-8') as fp:
    install_requires = fp.read()

setup(
    name='meep_mod',
    packages=find_packages(),
    version='1.0.0',
    author='yeastpowder',
)
