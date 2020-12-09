from setuptools import setup
from setuptools import find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="automol",
    version="0.0.1",
    author="Daniel Azanov, Felix Ning, Matthias Schniewind",
    author_email="matthias.schniewind@kit.edu",
    description="Automate ML on molecular data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    install_requires=["numpy", "pandas", "rdkit"],
    packages=find_packages(),
    include_package_data=True,
    package_data={"automol": ["*.json", "*.yaml"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    keywords=["machine", "learning", "molecular", "automated"]
)
