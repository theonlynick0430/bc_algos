from setuptools import setup, find_packages

setup(
    name="bc_algos",
    packages=[
        package for package in find_packages() if package.startswith("bc_algos")
    ],
)
