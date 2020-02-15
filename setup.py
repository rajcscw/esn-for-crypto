from setuptools import setup, find_packages

setup(
    name = "esn_for_crypto",
    version="0.1",
    packages=find_packages(),
    install_requires=["numpy", "sklearn", "networkx", "matplotlib", "scipy==1.1.0", "pillow"],
    python_requires=">3.7"
)