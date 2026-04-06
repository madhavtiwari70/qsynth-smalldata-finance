from setuptools import setup, find_packages

setup(
    name="iqp-finance-synth",
    version="0.1.0",
    description="Quantum Generative Models for Financial Synthetic Data using IQP Circuits",
    author="Your Name",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.23",
        "pandas>=1.5",
        "scikit-learn>=1.1",
        "matplotlib>=3.5",
        "seaborn>=0.12",
        "jax>=0.4",
        "jaxlib>=0.4",
        "pennylane>=0.35",
        "iqpopt>=0.1",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
    ],
)
