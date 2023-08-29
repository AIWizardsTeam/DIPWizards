from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='dipwizards',
    version='0.1',
    packages=find_packages(),
    install_requires=requirements,
    description='Dip package for images ',
    author='AI Wizards Team',
    author_email='aiwizardsteam@gmail.com',

    )
