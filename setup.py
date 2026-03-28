from setuptools import setup, find_packages

setup(
    name="ckd_experiments",
    version="1.0.0",
    description="Treatment Effect Estimation in Small-Animal Longitudinal Studies",
    author="",
    author_email="",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "openpyxl>=3.1.0",
    ],
    extras_require={
        "deep learning": [
            "torch>=2.0.0",
        ],
        "gpu": [
            "torch>=2.0.0",
            "cuda-enabled",
        ],
    },
)
