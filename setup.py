'''
Python implementation of ALI feature based SeqSLAM
'''
import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

# This is a plug-in for setuptools that will invoke py.test
# when you run python setup.py test
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest  # import here, because outside the required eggs aren't loaded yet
        sys.exit(pytest.main(self.test_args))

version = "0.1"

setup(name="ALI-Seq",
      version=version,
      description="Python implementation of SeqSLAM with ALI feature.",
      long_description=open("README.md").read(),
      classifiers=[ # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 1 - Planning',
          'Programming Language :: Python'
      ],
      keywords="place recognition, ALI-feature, SeqSLAM", # Separate with spaces
      author="maxtom",
      author_email="hitmaxtom@gmail.com",
      url="",
      license="MIT",
      packages=find_packages(exclude=['examples', 'tests']),
      include_package_data=True,
      zip_safe=True,
      tests_require=['pytest'],
      cmdclass={'test': PyTest},
      
      # TODO: List of packages that this one depends upon:   
      install_requires=['scipy', 'numpy', 'matplotlib', 'Pillow'],
      # TODO: List executable scripts, provided by the package (this is just an example)
      entry_points={
          'console_scripts': 
          ['pySeqSLAM=pyseqslam:main']
      }
)
