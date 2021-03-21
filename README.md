# AutoMol: AutoML for chemistry/materials

## Description
*	Development of an automated workflow to train ML models for typical chemistry datasets
*	Various features: fingerprints, rdkit, ...
*	Various sklearn models: RandomForest, GaussianProcess, GradientBoosting, ...
*	Interface to hyperparameter-optimization: via hyper_param_grid parameter
*	Modular: Extendible in a flexible way to generate new features and add new models

```python
from automol.pipeline import Pipeline
pipeline = Pipeline('<my_config_yaml_path>')
pipeline.train()
pipeline.get_statistics()
```

## Technical requirements
*	Save/Load procedure: TODO
*	Encapsulation: mlflow
*	Easy (re-)use
*	Python based
*	Makes use of ScikitLearn
*	Rdkit dependence wherever needed, but as little as possible: currently only in feature_generators.py to generate rdkit features
*	Automated plotting: currently only performance of models, hyper param scores and learning curve
*	Analysis of results: TODO
*	Handling of CV and test/train/validation splits, especially with small datasets: train_test_splits parameter for train/test splits, cv parameter is currently only for hyper param and learning curve

## How to use
                                   Config.yml


dataset_location: '../data/dsgdb9nsd'

dataset_class: 'QM9'

label: 'homo'

models_filter:

 - whitelist: 1

    git_uri: sklearn

    model_names: ['RandomForestRegressor']

problem: regression

amount: 200

features: ['fingerprint', 'rdkit']

dataset_split_test_size: 0.1

mlflow_experiment: qm9_dataset_automol_demo

train_test_splits: 2

cv: 5

hyper_param_grid:

    RandomForestRegressor: {'max_depth': [3, 5, 10]}

is_learning_curve: True

pca_preprocessing:

 model_name: 'MLPRegressor'

 feature: fingerprint

    n_components: 50

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
