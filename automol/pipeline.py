from __future__ import annotations
import yaml
import pandas as pd
import mlflow
from mlflow_utils.load_env import export_env
from sklearn.model_selection import train_test_split
from automol.datasets import Dataset
from automol.models import generate_all_possible_models
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt


class Pipeline:

    def __init__(self, input_yaml_file: str):
        with open(input_yaml_file, 'r') as file:
            try:
                self.spec = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise e
        self.data_set = Dataset.from_spec(self.spec)
        self.statistics = pd.DataFrame(columns=['model', 'feature', 'split_index', 'training_mae', 'training_mse',
                                                'training_r2_score', 'test_mae', 'test_mse', 'test_r2_score'])
        self.cv_results = {}
        self.pca_spec = self.spec['pca_preprocessing'] if 'pca_preprocessing' in self.spec else None
        self.models = generate_all_possible_models(self.spec['problem'], self.spec['models_filter'])
        self.hyper_param_grid = self.spec['hyper_param_grid'] if 'hyper_param_grid' in self.spec else {}
        self.feature_names = self.spec['features']
        self.label_name = self.spec['label']
        self.dataset_split_test_size = self.spec['dataset_split_test_size']
        self.train_test_splits = self.spec['train_test_splits']
        self.sh_split = ShuffleSplit(n_splits=self.train_test_splits, test_size=self.dataset_split_test_size)
        self.cv = self.spec['cv'] if 'cv' in self.spec else None

    def get_next_split_index(self, feature, label):
        return next(self.sh_split.split(feature, label))

    def mlflow_setup(self):
        export_env()
        mlflow.set_experiment(self.spec['mlflow_experiment'])

    def add_model_statistics(self, model_name, feature_name, model_statistics, split_index):
        model_statistics['feature'] = feature_name
        model_statistics['model'] = model_name
        model_statistics['split_index'] = int(split_index)
        self.statistics = self.statistics.append(model_statistics, ignore_index=True)

    def add_cv_results(self, model_name, feature_name, cv_results):
        if model_name in self.cv_results:
            if feature_name in self.cv_results[model_name]:
                self.cv_results[model_name][feature_name].append(cv_results)
            else:
                self.cv_results[model_name][feature_name] = [cv_results]
        else:
            self.cv_results[model_name] = {feature_name: [cv_results]}

    def train(self):
        self.mlflow_setup()
        label = self.data_set.get_feature(self.label_name)
        feature_pca = None
        for feature_name in self.feature_names:
            feature = self.data_set.get_feature(feature_name)
            if self.pca_spec and feature_name == self.pca_spec['feature']:
                feature_pca = self.data_set.get_feature_preprocessed_by_pca(feature_name, self.pca_spec['n_components'])
            self.train_with_feature(feature_name, feature, label, feature_pca)

    def train_with_feature(self, feature_name, feature, label, feature_pca=None):
        x_train_pca = x_test_pca = None
        for split_index in range(self.train_test_splits):
            train_index, test_index = self.get_next_split_index(feature, label)
            x_train, x_test = feature[train_index], feature[test_index]
            y_train, y_test = label[train_index], label[test_index]
            if self.pca_spec and feature_name == self.pca_spec['feature']:
                x_train_pca, x_test_pca = feature_pca[train_index], feature_pca[test_index]
            self.train_with_dataset_split(split_index, feature_name, x_train, y_train, x_test, y_test,
                                          x_train_pca, x_test_pca)

    def train_with_dataset_split(self, split_index, feature_name, x_train, y_train, x_test, y_test,
                                 x_train_pca=None, x_test_pca=None):

        for model in self.models:
            model_name = model.get_model_name()
            if self.pca_spec and feature_name == self.pca_spec['feature'] \
                    and model_name == self.pca_spec['model_name']:
                n_components = self.pca_spec['n_components']
                print(f'Running model {model_name} with feature {feature_name} '
                      f'(PCA to {n_components} dimensions).')
                self.run_model(model_name, model, x_train_pca, y_train, x_test_pca, y_test)
            else:
                print(f'Running model {model_name} with feature {feature_name}.')
                self.run_model(model_name, model, x_train, y_train, x_test, y_test)
            self.add_model_statistics(model_name, feature_name, model.get_statistics(), split_index)
            self.add_cv_results(model_name, feature_name, model.get_param_search_cv_results())

    def run_model(self, model_name, model, x_train, y_train, x_test, y_test, drop_nan=True):
        param_grid = self.hyper_param_grid[model_name] if model_name in self.hyper_param_grid else None
        if drop_nan:
            nan_x_train = np.argwhere(np.isnan(x_train)).flatten()[::len(x_train.shape)]
            nan_y_train = np.argwhere(np.isnan(y_train)).flatten()[::len(y_train.shape)]
            nan_train = np.unique(np.append(nan_x_train, nan_y_train))
            nan_x_test = np.argwhere(np.isnan(x_test)).flatten()[::len(x_test.shape)]
            nan_y_test = np.argwhere(np.isnan(y_test)).flatten()[::len(y_test.shape)]
            nan_test = np.unique(np.append(nan_x_test, nan_y_test))
            model.run(np.delete(x_train, nan_train, axis=0), np.delete(y_train, nan_train, axis=0),
                      np.delete(x_test, nan_test, axis=0), np.delete(y_test, nan_test, axis=0),
                      param_grid, self.cv)
        else:
            model.run(x_train, y_train, x_test, y_test, param_grid, self.cv)

    def get_statistics(self):
        return self.statistics

    def get_cv_results(self, model_name, feature_name, split_index):
        return self.cv_results[model_name][feature_name][split_index]

    def print_spec(self):
        print(yaml.dump(self.spec))

    def print_models(self):
        print('Models to run: {}'.format([model.get_model_name() for model in self.models]))

    def plot_grid_search(self, model_name, feature_name, grid_param_name):
        grid_param = self.hyper_param_grid[model_name][grid_param_name]
        _, ax = plt.subplots(1, 1)
        for split_index in range(self.train_test_splits):
            mean_test_score = self.get_cv_results(model_name, feature_name, split_index)['mean_test_score']
            std_test_score = self.get_cv_results(model_name, feature_name, split_index)['std_test_score']
            plt.errorbar(grid_param, mean_test_score, yerr=std_test_score,
                         fmt='-o', label=f'split_index = {split_index}')
        ax.set_title(f"Grid Search Scores for {model_name}\n", fontsize=16, fontweight='bold')
        ax.set_xlabel(grid_param_name, fontsize=16)
        ax.set_ylabel('CV mean_test_score', fontsize=16)
        ax.legend(loc="best", fontsize=15)
        ax.grid('on')
