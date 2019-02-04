#! /usr/bin/env python

# third party
from setuptools import find_packages, setup

setup(
    name='hsr-env',
    version='0.0.0',
    description='An environment conforming to OpenAI Gym standard\
            for the Toyota Human Support Robot',
    url='https://github.com/lobachevzky/hsr-env',
    author='Ethan Brooks',
    author_email='ethanabrooks@gmail.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
    packages=find_packages(),
    package_data={
        'environments': ['**/*.xml', '**/*.mjcf', 'hsr_meshes/meshes/*/*.stl'],
    },
    install_requires=['gym', 'numpy'])
