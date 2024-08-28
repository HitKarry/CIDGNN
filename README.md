# Official open-source code of "Dynamic graph neural network based on cross interaction for multivariable spatiotemporal graph data generation"

Multivariate spatiotemporal graph data generation technology is crucial for the construction of virtual environments in the metaverse. Existing methods struggle with temporal fluctuations and heterogeneity, limiting their ability to model key patterns across variables, time, and spatial dimensions. CIDGNN is a novel dynamic GNN designed to enhance the generation of multivariate spatiotemporal graph data by optimizing cross variable, cross temporal scale, and cross spatial node interactions. Specifically, utilizing adaptive multiscale identifier to handle unexpected noise in the time dimension; Introducing a heterogeneous information dynamic fusion module to dynamically fuse the multivariate feature at different nodes; Cross interaction dynamic GNN were proposed to extract scales with clearer trends and weaker noise, while fully utilizing the homogeneity and heterogeneity between nodes. Experiments shown that CIDGNN has higher performance, stronger noise resistance, and good robustness than existing methods, which can meet the requirements of high fidelity in metaverse construction.

# mgcpy

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

'CIDGNN' is a dynamic graph neural network model capable of handling high and multivariable data. The open source data contains the neural network, test framework, sample input data, and corresponding expected output.

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Setting up the development environment](#setting-up-the-development-environment)
- [License](#license)
- [Issues](https://github.com/neurodata/mgcpy/issues)

# Overview
``mgcpy`` aims to be a comprehensive independence testing package including commonly used independence tests and additional functionality such as two sample independence testing and a novel random forest-based independence test. These tests are not only included to benchmark MGC but to have a convenient location for users if they would prefer to utilize those tests instead. The package utilizes a simple class structure to enhance usability while also allowing easy extension of the package for developers. The package can be installed on all major platforms (e.g. BSD, GNU/Linux, OS X, Windows)from Python Package Index (PyPI) and GitHub.

# Documenation
The official documentation with usage is at: https://mgc.neurodata.io/
ReadTheDocs: https://mgcpy.readthedocs.io/en/latest/

# System Requirements
## Hardware requirements
`mgcpy` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for *macOS* and *Linux*. The package has been tested on the following systems:
+ macOS: Mojave (10.14.1)
+ Linux: Ubuntu 16.04

### Python Dependencies
`mgcpy` mainly depends on the Python scientific stack.

```
numpy
scipy
Cython
scikit-learn
pandas
seaborn
```

# Installation Guide:

### Install from PyPi
```
pip3 install mgcpy
```

### Install from Github
```
git clone https://github.com/neurodata/mgcpy
cd mgcpy
python3 setup.py install
```
- `sudo`, if required
- `python3 setup.py build_ext --inplace  # for cython`, if you want to test in-place, first execute this

# Setting up the development environment:
- To build image and run from scratch:
  - Install [docker](https://docs.docker.com/install/)
  - Build the docker image, `docker build -t mgcpy:latest .`
    - This takes 10-15 mins to build
  - Launch the container to go into mgcpy's dev env, `docker run -it --rm --name mgcpy-env mgcpy:latest`
- Pull image from Dockerhub and run:
  - `docker pull tpsatish95/mgcpy:latest` or `docker pull tpsatish95/mgcpy:development`
  - `docker run -it --rm -p 8888:8888 --name mgcpy-env tpsatish95/mgcpy:latest` or `docker run -it --rm -p 8888:8888 --name mgcpy-env tpsatish95/mgcpy:development`


- To run demo notebooks (from within Docker):
  - `cd demos`
  - `jupyter notebook --ip 0.0.0.0 --no-browser --allow-root`
  - Then copy the url it generates, it looks something like this: `http://(0de284ecf0cd or 127.0.0.1):8888/?token=e5a2541812d85e20026b1d04983dc8380055f2d16c28a6ad`
  - Edit this: `(0de284ecf0cd or 127.0.0.1)` to: `127.0.0.1`, in the above link and open it in your browser
  - Then open `mgc.ipynb`

- To mount/load local files into docker container:
  - Do `docker run -it --rm -v <local_dir_path>:/root/workspace/ -p 8888:8888 --name mgcpy-env tpsatish95/mgcpy:latest`, replace `<local_dir_path>` with your local dir path.
  - Do `cd ../workspace` when you are inside the container to view the mounted files. The **mgcpy** package code will be in `/root/code` directory.


## MGC Algorithm's Flow
![MGCPY Flow](https://raw.githubusercontent.com/neurodata/mgcpy/master/MGCPY.png)

## Power Curves
- Recreated Figure 2 in https://arxiv.org/abs/1609.05148, with the addition of MDMR and Fast MGC
![Power Curves](https://raw.githubusercontent.com/neurodata/mgcpy/master/power_curves_dimensions.png)

# License

This project is covered under the **Apache 2.0 License**.
