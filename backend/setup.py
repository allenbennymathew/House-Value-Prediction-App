from setuptools import setup, find_packages

setup(
    name="housing",
    version="0.2",
    description="Housing price prediction model",
    author="Allen",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy==2.0.2",
        "pandas==2.3.3",
        "scikit-learn==1.6.1",
        "six==1.17.0"
    ],
)
