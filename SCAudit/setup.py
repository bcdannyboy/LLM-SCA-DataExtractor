"""
Setup configuration for SCAudit package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="scaudit",
    version="1.0.0",
    author="SCAudit Team",
    description="Special Characters Attack auditing tool for LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scaudit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "click>=8.0.0",
        "langchain>=0.1.0",
        "langchain-community>=0.1.0",
        "langchain-openai>=0.1.0",
        "openai>=1.0.0",
        "anthropic>=0.18.0",
        "sqlalchemy>=2.0.0",
        "pysqlcipher3>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "faiss-cpu>=1.7.0",
        "tenacity>=8.0.0",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "pylint>=2.17.0",
            "mypy>=1.0.0",
        ],
        "gpu": [
            "faiss-gpu>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "scaudit=SCAudit.cli:cli",
        ],
    },
)