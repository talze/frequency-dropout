from setuptools import setup, find_packages

setup(
    name="frequency-dropout",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
    ],
    author="Original paper authors: T. Zeevi, R. Venkataraman, L. H. Staib, J. A. Onofrey",
    author_email="",
    description="Frequency domain dropout for uncertainty estimation via Monte Carlo sampling",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/frequency-dropout",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
)
