import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='mcabc',
    version='0.0.1',
    description='Model Comparison in Approximate Bayesian Computation',
    url='https://github.com/janfb/mcabc',
    author='janfb',
    packages=['mcabc', 'mcabc.mdn', 'mcabc.model', 'mcabc.tests', 'mcabc.utils'],
    license='MIT',
    long_description=read('README'),
    install_requires=['numpy', 'scipy', 'tqdm', 'multiprocessing', 'torch', 'random'],
    dependency_links=[]
)
