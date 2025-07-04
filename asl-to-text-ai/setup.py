"""
Setup script for ASL-to-Text AI package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="asl-to-text-ai",
    version="1.0.0",
    author="ASL-AI Team",
    author_email="support@asl-ai.com",
    description="Advanced AI system for translating American Sign Language into written text",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Flashinl/asl-to-text-ai",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video :: Capture",
        "Topic :: Communications :: Chat",
        "Topic :: Education",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "pre-commit>=3.3.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.13.0",
            "torch-gpu>=2.0.0",
        ],
        "deployment": [
            "gunicorn>=21.0.0",
            "docker>=6.1.0",
            "nginx",
        ]
    },
    entry_points={
        "console_scripts": [
            "asl-ai-server=web_app.app:main",
            "asl-ai-train=src.training.train_model:main",
            "asl-ai-test=tests.run_tests:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml", "*.txt", "*.md"],
        "web_app": ["templates/*", "static/*/*"],
        "data": ["vocabulary/*", "models/*"],
    },
    zip_safe=False,
    keywords=[
        "asl",
        "sign-language",
        "ai",
        "machine-learning",
        "computer-vision",
        "translation",
        "accessibility",
        "real-time",
        "tensorflow",
        "pytorch",
        "mediapipe",
        "opencv",
        "flask",
        "websocket"
    ],
    project_urls={
        "Bug Reports": "https://github.com/Flashinl/asl-to-text-ai/issues",
        "Source": "https://github.com/Flashinl/asl-to-text-ai",
        "Documentation": "https://github.com/Flashinl/asl-to-text-ai/docs",
        "Funding": "https://github.com/sponsors/Flashinl",
    },
)
