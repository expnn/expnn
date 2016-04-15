from setuptools import setup
from setuptools import find_packages


setup(name='expnn',
      version='0.1',
      description='Theano-based Neural Network Library',
      author='Yunchuan Chen',
      author_email='chenych11@gmail.com',
      url='https://github.com/chenych11/expnn',
      download_url='https://github.com/chenych11/expnn',
      license='MIT',
      install_requires=['theano', 'pyyaml', 'h5py', 'six'],
      packages=find_packages())
