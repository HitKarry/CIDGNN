# Official open-source code of "Dynamic graph neural network based on cross interaction for multivariable spatiotemporal graph data generation"

Multivariate spatiotemporal graph data generation technology is crucial for the construction of virtual environments in the metaverse. Existing methods struggle with temporal fluctuations and heterogeneity, limiting their ability to model key patterns across variables, time, and spatial dimensions. CIDGNN is a novel dynamic GNN designed to enhance the generation of multivariate spatiotemporal graph data by optimizing cross variable, cross temporal scale, and cross spatial node interactions. Specifically, utilizing adaptive multiscale identifier to handle unexpected noise in the time dimension; Introducing a heterogeneous information dynamic fusion module to dynamically fuse the multivariate feature at different nodes; Cross interaction dynamic GNN were proposed to extract scales with clearer trends and weaker noise, while fully utilizing the homogeneity and heterogeneity between nodes. Experiments shown that CIDGNN has higher performance, stronger noise resistance, and good robustness than existing methods, which can meet the requirements of high fidelity in metaverse construction.

# CIDGNN

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

'CIDGNN' is a dynamic graph neural network model capable of handling high and multivariable data. The open source data contains the neural network, test framework, sample input data, and corresponding expected output.

- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Setting up the development environment](#setting-up-the-development-environment)
- [License](#license)


# System Requirements
## Hardware requirements
`CIDGNN` requires a standard computer with enough RAM to support in-memory operations and a high-performance GPU to support fast operations on high-dimensional data.

## Software requirements
### OS Requirements
This package is supported for *Windows* and *Linux*. The package has been tested on the following systems:
+ Windows: Windows 10 22H2
+ Linux: Ubuntu 16.04

### Python Dependencies
`CIDGNN` mainly depends on the Python scientific stack.

```
einops==0.8.0
fbm==0.3.0
matplotlib==3.7.2
numpy==1.24.3
pandas==2.0.3
pmdarima==2.0.4
ptflops==0.7.3
pynvml==11.5.3
scikit_learn==1.5.1
scipy==1.10.1
seaborn==0.13.2
sympy==1.12
torch==2.3.1
torch_cluster==1.6.3
tqdm==4.66.4
tvm==1.0.0
xarray==2022.11.0
```

# License

This project is covered under the **Apache 2.0 License**.
