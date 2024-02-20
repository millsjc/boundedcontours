from setuptools import setup, find_packages

setup(
    name="boundedcontours",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)
