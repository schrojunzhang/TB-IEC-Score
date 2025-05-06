from setuptools import setup, find_packages

setup(
    name="tb_iecs",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.18.0",
        "pandas>=1.0.0",
        "scikit-learn>=0.22.0",
        "xgboost>=1.0.0",
        "hyperopt>=0.2.5",
        "joblib>=0.14.0",
        "matplotlib>=3.1.0",
        "seaborn>=0.10.0",
        "imbalanced-learn>=0.6.0",
    ],
    entry_points={
        "console_scripts": [
            "tb_iecs=cli:main",
        ],
    },
    author="Xujun Zhang",
    author_email="",
    description="TB-IEC-Score: An accurate and efficient machine learning-based scoring function for virtual screening",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/username/TB-IEC-Score",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
) 