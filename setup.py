from setuptools import setup, find_packages


with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="treeqinetic",
    version="1.0.0",
    packages=find_packages(),
    install_requires=requirements,
    author="Kyell Jensen",
    author_email="mail@kyelljensen.de",
    description="treeqinetic Beschreibung",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kyelljensen/linescale3",
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: Apache Software License",
    ]
)

