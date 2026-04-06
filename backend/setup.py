from setuptools import setup, find_packages

setup(
    name="housing",
    version="0.2",
    description="Housing price prediction model",
    author="Allen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
)
