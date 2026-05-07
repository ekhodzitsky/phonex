"""Setup script for phonex Python bindings."""

from setuptools import setup

setup(
    name="phonex",
    version="0.2.1",
    description="Python bindings for phonex — on-device speech-to-text",
    py_modules=["phonex"],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
