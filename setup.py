from setuptools import setup, find_packages

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


if __name__ == "__main__":
    setup(
        name="oe3dis",
        version="1.0",
        description="oe3dis",
        author="Phuc Nguyen",
        packages=find_packages(include=['OmniScientModel']),
    )
