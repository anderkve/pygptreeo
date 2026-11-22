import setuptools

setuptools.setup(
    name='pygptreeo',
    version='0.1.0',
    author='GPTreeO Project Contributors',
    author_email='anders.kvellestad@fys.uio.no',
    description='A Python package for online regression using a dynamic tree of Gaussian Process regressors.',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    url='http://github.com/anderkve/pygptreeo',
    packages=setuptools.find_packages(include=['pygptreeo', 'pygptreeo.*']),
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'binarytree',
        'tqdm',
    ],
    extras_require={
        'gpytorch': ['gpytorch', 'torch'],  # For GPyTorch backend support
        'neural': ['torch'],  # For the feature network global model (global_mean='net')
        'all': ['gpytorch', 'torch'],  # Install all optional dependencies
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)
