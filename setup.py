from setuptools import setup, find_packages

# del build -Recurse; del dist -Recurse; del sp_test.egg-info -Recurse
# py setup.py sdist bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.22.0",
                "pandas>=1.3.0",
                "yfinance>=0.1.74",
                "matplotlib>=3.4.2",
                "mplfinance>=0.12.9b1",
                "fpdf>=1.7.2",
                "seaborn>=0.11.2",
                "scipy>=1.7.0"]

setup(name="ndxtest", version="0.0.3",
      author="lksdnx",
      author_email="lukas.dnx@gmail.com",
      description="A python newbies package to test trading strategies on the S&P 500.",
      long_description=readme,
      long_description_content_type="text/markdown",
      url="https://github.com/lksndx/spy/",
      packages=find_packages(),
      install_requires=requirements,
      classifiers=["Programming Language :: Python :: 3.9", "License :: OSI Approved :: MIT License"]
      )
