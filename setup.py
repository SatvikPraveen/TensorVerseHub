# Location: /setup.py

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="tensorversehub",
    version="1.0.0",
    author="TensorVerseHub Contributors",
    author_email="contact@tensorversehub.com",
    description="Comprehensive TensorFlow learning hub with 23+ hands-on notebooks and production utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SatvikPraveen/TensorVerseHub",
    project_urls={
        "Bug Tracker": "https://github.com/SatvikPraveen/TensorVerseHub/issues",
        "Documentation": "https://tensorversehub.readthedocs.io/",
        "Source Code": "https://github.com/SatvikPraveen/TensorVerseHub",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Framework :: TensorFlow",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.9.0",
            "flake8>=6.1.0",
            "mypy>=1.6.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.4.0",
            "sphinx>=7.2.0",
        ],
        "serving": [
            "flask>=3.0.0",
            "streamlit>=1.28.0",
            "fastapi>=0.104.0",
            "uvicorn>=0.24.0",
        ],
        "gpu": [
            "tensorflow-gpu>=2.13.0,<2.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tensorverse-train=scripts.train_models:main",
            "tensorverse-evaluate=scripts.evaluate_models:main",
            "tensorverse-convert=scripts.convert_models:main",
            "tensorverse-serve=examples.serving_examples.flask_tensorflow_api:main",
        ],
    },
    include_package_data=True,
    package_data={
        "tensorversehub": [
            "data/**/*",
            "models/**/*",
            "docs/**/*",
        ],
    },
    keywords="tensorflow keras machine-learning deep-learning computer-vision nlp generative-models model-optimization",
    zip_safe=False,
)