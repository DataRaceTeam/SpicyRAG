from setuptools import find_packages, setup


def read_requirements():
    with open("requirements.txt") as f:
        return f.read().splitlines()


setup(
    name="interface",
    version="0.1.0",
    description="A description of your package",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Your Name",
    packages=find_packages(exclude=["tests*"]),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
