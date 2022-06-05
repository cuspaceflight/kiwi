"""
Allows installation via pip, e.g. by navigating to this directory with the command prompt, and using 'pip install .'
"""

from setuptools import setup, find_packages

setup(
    name='cusfkiwi',
    author = 'Daniel Gibbons',       
    version='0.1.0',
    packages = find_packages(),
    install_requires=['numpy'],
    license = '	AGPL-3.0',
    author_email='daniel.u.gibbons@gmail.com',
    description='Generalised 6DOF trajectory and dynamics propagator'
)
