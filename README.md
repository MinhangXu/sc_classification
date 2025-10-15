# sc_classification
Modular repo for developing  single cell level classification from dimension reduction to semi-/supervised classification algorithm to downstream validations 

## Project Structure

This project follows a standard `src` layout to cleanly separate the installable Python library from run scripts and analysis notebooks.

*   `src/sc_classification/`: This is the core Python package. It contains all the reusable modules for the pipeline, such as preprocessing, dimensionality reduction, and classification methods. Code in here is meant to be imported, not run directly.

*   `scripts/`: This directory contains standalone Python scripts that serve as entry points for running experiments. These scripts import the `sc_classification` package to execute the pipeline with specific configurations.

*   `notebooks/`: This directory is for Jupyter notebooks used for exploratory data analysis, visualization, and interpretation of results.

*   `DESIGN.md`: Provides a high-level overview of the software architecture and pipeline workflow. 
