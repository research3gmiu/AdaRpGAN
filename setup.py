"""Fallback setup.py for older pip versions that don't support pyproject.toml."""
from setuptools import setup, find_packages

# Read the project-level README for long_description
try:
    with open("adarppgan/README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = (
        "AdaRpGAN: A Self-Tuning GAN Training Framework "
        "with Adaptive Gradient Penalty and Dynamic Critic Scheduling"
    )

setup(
    name="adarppgan",
    version="1.0.0",
    description=(
        "AdaRpGAN: A Self-Tuning GAN Training Framework "
        "with Adaptive Gradient Penalty and Dynamic Critic Scheduling"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="AdaRpGAN Authors",
    license="MIT",
    url="https://github.com/yourusername/adarppgan",
    project_urls={
        "Documentation": "https://github.com/yourusername/adarppgan#readme",
        "Source": "https://github.com/yourusername/adarppgan",
        "Issues": "https://github.com/yourusername/adarppgan/issues",
    },
    packages=find_packages(include=["adarppgan", "adarppgan.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "wandb": ["wandb>=0.15.0"],
        "pdf": ["fpdf2>=2.7.0"],
        "all": ["wandb>=0.15.0", "tensorboard>=2.13.0", "fpdf2>=2.7.0"],
    },
    entry_points={
        "console_scripts": [
            "adarppgan=adarppgan.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=[
        "GAN", "generative-adversarial-network",
        "deep-learning", "pytorch", "adaptive-training",
    ],
)
