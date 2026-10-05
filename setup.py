from setuptools import setup, find_packages

setup(
    name="6g-network-slicing",
    version="1.0.0",
    description="6G Network Slicing Platform with DRL and XAI",
    author="KasuSathvikaMary",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "torch>=2.0.0",
        "fastapi>=0.100.0",
        "uvicorn>=0.23.0",
        "pydantic>=2.0.0",
        "simpy>=4.0.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "pyyaml>=6.0",
        "aiohttp>=3.8.0",
        "prometheus-client>=0.17.0",
        "python-jose[cryptography]>=3.3.0",
        "passlib>=1.7.4",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "6g-slicing-api=api.main:app",
        ]
    },
    python_requires=">=3.9",
)
