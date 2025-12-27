import setuptools

# Using the more comprehensive README content planned in the previous subtask
# as the one read from file was minimal.
long_description_content = """
"""

setuptools.setup(
    name='pygptreeo',
    version='0.1.0',
    author='GPTreeO Project Contributors',
    author_email='gptreeo-dev@example.com',
    description='A Python package for online regression using a dynamic tree of Gaussian Process regressors.',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    url='http://github.com/user/pygptreeo', # Placeholder URL
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'binarytree',
        'tqdm',
        'joblib',
    ],
    extras_require={
        'gpytorch': ['gpytorch', 'torch'],  # For GPyTorch backend support
        'all': ['gpytorch', 'torch'],  # Install all optional dependencies
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)', # Updated based on LICENSE file
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
