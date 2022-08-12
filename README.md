<div id="top"></div>


<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]




<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/lksdnx/ndxtest">
    <img src="https://github.com/lksdnx/ndxtest/blob/master/docs/images/ndxtest_logo.png?raw=true" alt="drawing" width="300"/>
  </a>

  <p align="center">
    <strong>lets you test trading strategies on the S&P 500.</strong>
    <br />
    <br />
    <strong>pip install ndxtest</strong>
    <br />
    <br />
    <a href="https://ndxtest.readthedocs.io/en/latest/"><strong>Explore the documentation</strong></a>
    <br />
    <br />
    <a href="https://github.com/lksdnx/ndxtest/issues">Report Bug</a>
    ·
    <a href="https://github.com/lksdnx/ndxtest/issues">Request Feature</a>
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

`ndxtest` is a package for Python developed by `lksdnx`. It lets you test trading ideas 
and perform backtests on the S&P 500. The daily price data for S&P 500 companies is available in the 
repository.

Developing, documenting, testing and releasing `ndxtest` was and is primarily an educational project. 
Use it at your own risk and read the <a href="https://ndxtest.readthedocs.io/en/latest/DISCLAIMER.html">disclaimer</a> 
on the documentation page.

<p align="right">(<a href="#top">back to top</a>)</p>


### Built With

[<img src="Python 3.9" height="40" src="https://www.python.org/static/img/python-logo.png" width="140"/>][Python-url]
[<img src="https://numpy.org/images/logo.svg" alt="drawing" width="50"/>][Numpy-url]

[<img src="https://pandas.pydata.org/static/img/pandas_white.svg" alt="drawing" width="100"/>][Pandas-url]
[<img src="https://matplotlib.org/_static/images/logo2.svg" alt="drawing" height="45" width="100"/>][Matplotlib-url]


<p align="right">(<a href="#top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

For detailed information on how to install and use `ndxtest` please refer to the 
<a href="https://ndxtest.readthedocs.io/en/latest/MANUAL.html">User Manual</a> on the documentation page.

### Prerequisites

- Install [Python 3.9](https://www.python.org/downloads/) or higher.
- Install an IDE of your choice to work with. 
- Install [git](https://git-scm.com/downloads) to easily download the required [data](https://github.com/lksdnx/ndxtest/tree/master/data) directory from the repository.
- **Allow both Python and git to be added to the ``Path``** during their respective installations.

### Installation

In the terminal do:

   ```sh
   pip install ndxtest
   ```

This will install `ndxtest` as well as its python package dependencies: 
- `numpy`, `pandas`, `matplotlib`, `yfinance` and `fpdf`

Then in the terminal do:

   ```sh
   git clone --depth 1 --filter=blob:none --sparse https://github.com/lksdnx/ndxtest
   cd ndxtest
   git sparse-checkout set data
   ```

You can delete the .git folder afterwards. Keep the `data` folder.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->
## Usage

For instructions on how to work with this package, please refer to
the [Documentation](https://ndxtest.readthedocs.io/en/latest/).


<!-- ROADMAP -->
## Roadmap

See <a href="https://ndxtest.readthedocs.io/en/latest/FUTUREFEATURES.html">here</a> about new features of upcoming releases of ``ndxtest``

See the [open issues](https://github.com/lksdnx/ndxtest/issues) for a list of proposed features (and known issues).

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Any contributions are **greatly appreciated**.

If you have an idea that would make ``ndxtest`` better, please fork the repo and create a pull request. 
You can also simply open an [issue](https://github.com/lksdnx/ndxtest/issues) .
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch: `git checkout -b feature/AmazingFeature`
3. Commit your Changes: `git commit -m 'Add some AmazingFeature'`
4. Push to the Branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

lksdnx - lukas.dnx@gmail.com

Project Link: [https://github.com/lksdnx/ndxtest](https://github.com/lksdnx/ndxtest)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

I express my gratitude to all the people who contributed to build the
[FreeCodeCamp](https://www.freecodecamp.org/learn/). You have created an amazing
online resource. Furthermore I would like to thank all the content creators on YouTube for producing excellent
educative material on coding. Keep up the great work!

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/lksdnx/ndxtest.svg?style=for-the-badge
[contributors-url]: https://github.com/lksdnx/ndxtest/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/lksdnx/ndxtest.svg?style=for-the-badge
[forks-url]: https://github.com/lksdnx/ndxtest/network/members
[stars-shield]: https://img.shields.io/github/stars/lksdnx/ndxtest.svg?style=for-the-badge
[stars-url]: https://github.com/lksdnx/ndxtest/stargazers
[issues-shield]: https://img.shields.io/github/issues/lksdnx/ndxtest.svg?style=for-the-badge
[issues-url]: https://github.com/lksdnx/ndxtest/issues
[license-shield]: https://img.shields.io/github/license/lksdnx/ndxtest.svg?style=for-the-badge
[license-url]: https://github.com/lksdnx/ndxtest/blob/master/LICENSE.txt

[Python]: https://www.python.org/static/img/python-logo.png
[Python-url]: https://www.python.org/
[Numpy-url]: https://numpy.org/
[Pandas]: https://pandas.pydata.org/static/img/pandas_white.svg
[Pandas-url]: https://pandas.pydata.org/
[Matplotlib]: https://matplotlib.org/stable/_static/logo2.svg
[Matplotlib-url]: https://matplotlib.org/

