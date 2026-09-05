from setuptools import find_packages, setup

setup(
    name="isaaclab_fpo",
    version="0.2.0",
    packages=find_packages(),
    description="FPO++ training stack vendored into unitree_rl_lab (official Unitree envs).",
    install_requires=["GitPython"],
)
