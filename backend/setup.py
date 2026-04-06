from setuptools import setup, find_packages

setup(
    name="housing",
    version="0.2",
    description="Housing price prediction model",
    author="Allen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy==1.26.4",
        "pandas==2.2.0",
        "scikit-learn==1.6.1",
        "six==1.17.0"
    ],
)
