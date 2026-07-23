# Official open-source code of "Cross-Interaction Dynamic Graph Neural Network for Heterogeneous Spatiotemporal Forecasting on Irregular Networks"

Forecasting heterogeneous variables on an irregular network requires a model to represent complex variable coupling, dynamic  temporal evolution at different scales, and state-dependent interaction among spatial nodes. Existing approaches often merge these dependencies in one attention or graph operator, or use relations that remain unchanged across input samples. We propose the Cross-Interaction Dynamic Graph Neural Network (CIDGNN) for deterministic multistep forecasting. CIDGNN first identifies dominant temporal scales and computes variable-fusion weights from the current input. It then performs message passing on separate temporal-scale and spatial-node graphs and adjusts the retained edges of both graphs according to the current state. Spatial propagation uses positive and negative branches for high-affinity and low-affinity neighbors. Training employs the Transformation-Invariant Loss with Distance Equilibrium and Homoscedastic Uncertainty (TILDE-HU), which combines pointwise error with temporal-shape consistency and learns weights for variables and loss terms with different numerical scales. Experiments on WeatherBench and CLDAS-V2.0 evaluate in-domain performance, component contributions, robustness to Gaussian input noise, predictive scalability on 100- and 512-node graphs, regional transfer after fine-tuning, and applicability after retraining on another atmospheric product. CIDGNN achieves the best RMSE, MAE, and ACC in all reported variable--metric--lead-time comparisons. Relative to the SOTA model TimeFilter, RMSE is reduced by 1.95%--5.07% on the primary WeatherBench benchmark, 4.24%--7.97% after regional fine-tuning, and 5.11%--7.30% on CLDAS-V2.0.

# CIDGNN

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

'CIDGNN' is a dynamic graph neural network model capable of handling high and multivariable data. The open source data contains the neural network, test framework, sample input data, and corresponding expected output.

- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Instructions to run on data](#Instructions-to-run-on-data)
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

# Instructions to run on data

Due to the large size of the data and weight files, we host them on other data platforms, please download the relevant data from the link below.

Input data：https://mega.nz/file/jIEAzAhI#_PWOKOwGBvpAF_yOpYe7uksy8LOmnZta6f2I55kk5fA

Output data：https://mega.nz/file/fRVm1T7S#SWHCbu2tkFpME3Y-Eh7LWz15aVV4yI-4U-ZAqi_QpdA

Weight data：https://mega.nz/file/2cdBULiB#boGkh154f_97hbpbFIzYHl7j7iVLh_93gAeNH46L0EQ

Put the downloaded data into the specified folder, execute main.py to run and generate the result data, and evaluate.py to automate the quantitative evaluation of the data to generate the result.

# License

This project is covered under the **Apache 2.0 License**.
