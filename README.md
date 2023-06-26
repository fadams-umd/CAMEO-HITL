This is the code repository for the paper ["Human-In-the-Loop for Bayesian Autonomous Materials Phase Mapping"](https://arxiv.org/abs/2306.10406) by...

Felix Adams (1) (ORCID: [0000-0001-5802-1072](https://orcid.org/0000-0001-5802-1072))

Austin McDannald (2) (ORCID: [0000-0002-3767-926X](https://orcid.org/0000-0002-3767-926X))

Ichiro Takeuchi (1) (ORCID: [0000-0003-2625-0553](https://orcid.org/0000-0003-2625-0553))

A. Gilad Kusne (1,2) (ORCID: [0000-0001-8904-2087](https://orcid.org/0000-0001-8904-2087))

1. Materials Science & Engineering Dept, University of Maryland, College Park MD
2. Materials Measurement Science Division, Material Measurement Laboratory, National Institute of Standards and Technology, Gaithersburg MD

# Repository Contents
The [`hitl_environment.yml`](hitl_environment.yml) file can be used to create a `conda` environment with all the libraries used for this project.

[`hitl.ipynb`](hitl.ipynb) is a Python Jupyter notebook with a demo of the Human-In-The-Loop technique and a link to download the dataset used in the paper. The dataset download is compressed and must be unzipped before use.

[`index_plot.py`](index_plot.py) contains the code which generated `Sample Indices.svg`, which displays the sample index for each composition in the dataset for convenience.

[`quantitative_results.ipynb`](quantitative_results.ipynb) contains the code which generated the samples used for the Mann - Whitney test (stored in `./quantitative_results/`).

[`create_figures.ipynb`](create_figure.ipynb) contains the code which generated rough versions of all the figures in the paper. 

[`compare_labels.py`](compare_labels.py), [`compare_variance.py`](compare_variance.py), [`gpc_test.ipynb`](gpc_test.ipynb), and `autonomous_predicted_labels.txt` are all tests created over the course of the project.

# Repository History
The default branch of this repository is the `entropy` branch, named for the acquisition function used. For a time, a different acquisition was used for a portion of the project, until the `entropy` branch was created.

