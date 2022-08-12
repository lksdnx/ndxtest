from setuptools import setup, find_packages

# del build -Recurse; del dist -Recurse; del ndxtest.egg-info -Recurse
# py setup.py sdist bdist_wheel

# uploading to test pypi
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*

# installing from test pypi
# pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple ndxtest

# uploading to pypi
# twine upload dist/*


with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = ["numpy>=1.21.0",
                "pandas>=1.3.0",
                "yfinance>=0.1.74",
                "matplotlib>=3.4.2",
                "fpdf>=1.7.2",
                "openpyxl>=3.0.7"]

# deleted sphinx from here

setup(name="ndxtest", version="0.0.1",
      author="lksdnx",
      author_email="lukas.dnx@gmail.com",
      description="ndxtest lets you test trading strategies on the S&P 500.",
      long_description=readme,
      long_description_content_type="text/markdown",
      url="https://github.com/lksndx/ndxtest/",
      packages=find_packages(),
      install_requires=requirements,
      classifiers=["Programming Language :: Python :: 3.9", "License :: OSI Approved :: MIT License"]
      )
