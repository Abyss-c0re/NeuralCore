import toml
from setuptools import setup, find_packages

# Load data from pyproject.toml
pyproject = toml.load("pyproject.toml")
project = pyproject["project"]

# Extract project metadata from the pyproject.toml
name = project["name"]
version = project["version"]
description = project["description"]
long_description = open(project["readme"]).read()  # Read the README file content
long_description_content_type = "text/markdown"
author = project["authors"][0]["name"]
author_email = project["authors"][0]["email"]
requires_python = project["requires-python"]
url = project.get("homepage", "https://github.com/yourusername/neuralcore")  # Use a default URL if missing

# Extract dependencies from the pyproject.toml
install_requires = project["dependencies"]

# Scripts section (if defined)
scripts = project.get("scripts", {})

setup(
    name=name,
    version=version,
    description=description,
    long_description=long_description,
    long_description_content_type=long_description_content_type,
    author=author,
    author_email=author_email,
    url=url,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=requires_python,
    scripts=scripts,  # Include any scripts defined in pyproject.toml
)