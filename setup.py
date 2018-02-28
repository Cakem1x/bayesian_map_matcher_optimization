from setuptools import setup

setup(name='bayropt',
      version='0.1',
      description='Bayesian optimization framework, adapted for optimizing robot perception algorithms, e.g. feature-based map matcher.',
      url='git@github.com:Cakem1x/bayesian_map_matcher_optimization.git',
      author='Matthias Holoch',
      author_email='mholoch@gmail.com',
      license='BSD-2-Clause',
      packages=['bayropt'],
      install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'scikit-learn',
          'bayesian-optimization',
          'pyyaml',
      ],
      zip_safe=True)
