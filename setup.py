from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "torchvision",
    "deltalake",
]

setup(
    name="torchdelta",
    packages=find_packages(exclude=["tests", "tests.*"]),
    package_data={"": ["*.yaml", "*.html"]},
    include_package_data=True,
    install_requires=INSTALL_REQUIRES,
    setup_requires=["wheel"],
    version="0.0.1",
    description="",
    author="",
)
