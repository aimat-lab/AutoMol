# automol: AutoML for chemistry/materials

## Description
*	Development of an automated workflow to train ML models for typical chemistry datasets
*	Various descriptors: FPs, features, …
*	Various models: rdkit, NNs, GCNs
*	Interface to hyperparameter-optimization (Topic 2)
*	Modular: Extendible in a flexible way

## Technical requirements
*	Save/Load procedure
*	Encapsulation (singularity/docker/mlflow)
*	Easy (re-)use
*	python based
*	Makes use of ScikitLearn, TensorFlow/Keras, and optionally PyTorch
*	rdkit dependence wherever needed, but as little as possible
*	Automated plotting and analysis of results
*	Handling of CV and test/train/validation splits, especially with small datasets

## Literature
*	Representations of 3D structures: https://iopscience.iop.org/article/10.1088/2632-2153/abb212/meta
*	Library that implements several representations and ML models: https://www.qmlcode.org
*	Review article with multiple representations and models in Figure 1: https://www.nature.com/articles/s41570-020-0189-9
*	RDKit library, to handle smiles codes and compute molecular features: https://www.rdkit.org/

## Datasets
*	QM9 (including coordinates): https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
*	Non fullerene acceptors (only smiles, graph): https://www.sciencedirect.com/science/article/pii/S2542435117301307

## Project documentation
https://docs.google.com/document/d/1rJ60CYRD7Ljd7DSjCcH7kWHn_QeHakgaioclfYlulls

## Conda recipe
Install conda-build:
```
$ conda install -c conda-forge conda-build
```
Build conda recipe:
```
$ conda build -c conda-forge conda_recipe/ --python=3
```
List linked packages to check if automol is built successfully:
```
$ conda list
```
Install conda package:
```
$ conda install --use-local automol
```
