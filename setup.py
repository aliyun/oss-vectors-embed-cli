#!/usr/bin/env python3
"""Setup script for OSS Vectors CLI."""

from setuptools import setup, find_packages

# Import version from the package
exec(open("oss_vectors/__version__.py").read())

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="oss-vectors-embed-cli",
    version=__version__,
    author="fx",
    author_email="",
    maintainer="fx",
    maintainer_email="",
    description="Standalone CLI for OSS Vector operations with DashScope embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aliyun/oss-vectors-embed-cli",
    project_urls={
        "Bug Reports": "https://github.com/aliyun/oss-vectors-embed-cli/issues",
        "Source": "https://github.com/aliyun/oss-vectors-embed-cli",
        "Documentation": "https://github.com/aliyun/oss-vectors-embed-cli#readme",
        "Homepage": "https://github.com/aliyun/oss-vectors-embed-cli",
    },
    packages=find_packages(),
    keywords="oss, vectors, embeddings, dashscope, cli, machine-learning, ai",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
    ],
    python_requires=">=3.9",
    install_requires=[
        "alibabacloud-oss-v2>=1.2.1",
        "dashscope>=1.25.1",
        "click>=8.0.0",
        "rich>=12.0.0",
        "pydantic>=1.10.0",
    ],
    entry_points={
        "console_scripts": [
            "oss-vectors-embed=oss_vectors.cli:main",
        ],
    },
)
