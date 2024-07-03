from setuptools import setup


def readme():
    with open("README.md") as readme_file:
        return readme_file.read()


configuration = {
    "name": "utrees",
    "version": "0.1.1",
    "description": "Unmasking trees for tabular data generation and imputation",
    "long_description": readme(),
    "classifiers": [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved",
        "Programming Language :: Python",
        "Topic :: Software Development",
        "Topic :: Scientific/Engineering",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    "keywords": "tabular,imputation,generation",
    "url": "http://github.com/calvinmccarter/unmasking-trees",
    "author": "Calvin McCarter",
    "author_email": "mccarter.calvin@gmail.com",
    "maintainer": "Calvin McCarter",
    "maintainer_email": "mccarter.calvin@gmail.com",
    "packages": ["utrees"],
    "install_requires": [
        "kditransform",
        "numba >= 0.48",
        "numpy",
        "scikit-learn >= 0.23",
        "scipy >= 1.0",
        "xgboost",
    ],
    "ext_modules": [],
    "cmdclass": {},
    "test_suite": "nose.collector",
    "tests_require": ["nose"],
    "data_files": (),
    "zip_safe": True,
}

setup(**configuration)
