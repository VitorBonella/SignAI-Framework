from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='signai',
    version='0.1.0',
    description='A package for vibration signal analysis using AI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VitorBonella/SignAI-Framework',
    author='Vitor Bonella',
    packages=find_packages(),
    install_requires=[]
)