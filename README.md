# mcap_utils
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains a src, tests, skeleton with common lintering, developing and testing tools.  

It is focused to use the pyproject.toml to prevent a larger amount of boilerplate-files to configure the whole project. Dependencies are managed with poetry.

As stated [here](https://stackoverflow.com/questions/62983756/what-is-pyproject-toml-file-for) [PEP 518](https://peps.python.org/pep-0518/#rationale) is introducing the pyproject.toml. [PEP 660](https://peps.python.org/pep-0660/) adds the functionality for editable installs which makes the setup.py irrelevant. 

With [PEP 631](https://peps.python.org/pep-0631/) it becomes easily to add optional-dependencies and the need for the setup.cfg is not relevant anylonger. 

# Additional installation steps
```bash
sudo apt-get install python3-tk
```