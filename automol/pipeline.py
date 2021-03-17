from __future__ import annotations
import yaml
import pandas as pd
import mlflow
from mlflow_utils.load_env import export_env
from automol.datasets import Dataset
from automol.models import generate_all_possible_models
from sklearn.model_selection import ShuffleSplit
import numpy as np
import matplotlib.pyplot as plt


class Pipeline:

    def __init__(self, input_yaml_file: str):
        """
        Initializing of all necessary attributes of a pipeline given a config file in yaml format.

        Args:
            input_yaml_file: Pipeline configuration of attributes like models, features, labels, etc.
        """
        with open(input_yaml_file, 'r') as file:
            try:
                self.spec = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise e
        self.data_set = Dataset.from_spec(self.spec)
        self.statistics = pd.DataFrame(columns=['model', 'feature', 'split_index', 'training_mae', 'training_mse',
                                                'training_r2_score', 'test_mae', 'test_mse', 'test_r2_score'])
        self.cv_results = self.learning_curve_data = {}
        self.pca_spec = self.spec['pca_preprocessing'] if 'pca_preprocessing' in self.spec else None
        self.models = generate_all_possible_models(self.spec['problem'], self.spec['models_filter'])
        self.model_names = [model.get_model_name() for model in self.models]
        self.hyper_param_grid = self.spec['hyper_param_grid'] if 'hyper_param_grid' in self.spec else {}
        self.feature_names = self.spec['features']
        self.label_name = self.spec['label']
        self.dataset_split_test_size = self.spec['dataset_split_test_size']
        self.train_test_splits = self.spec['train_test_splits']
        self.dataset_splitter = ShuffleSplit(n_splits=self.train_test_splits, test_size=self.dataset_split_test_size)
        self.cv = self.spec['cv'] if 'cv' in self.spec else None
        self.mlflow_experiment = self.spec['mlflow_experiment']
        self.is_learning_curve = 'is_learning_curve' in self.spec

    def get_next_train_test_indices(self, feature, label):
        """

        Args:
            feature: feature data to split
            label: label data to split

        Returns: indices train_index, test_index based on the dataset splitter

        """
        return next(self.dataset_splitter.split(feature, label))

    def mlflow_setup(self):
        """

        Method to execute necessary steps for mlflow setup

        """
        export_env()
        mlflow.set_experiment(self.mlflow_experiment)

    def add_model_statistics(self, model_name, feature_name, model_statistics, split_index):
        """
        Saves model statistics for later use in a pipeline attribute

        Args:
            model_name: model name without arguments
            feature_name: feature name without arguments
            model_statistics: scores like mae, mse for respective feature and train test split
            split_index: index of a train test split

        """
        if model_statistics is None:
            return
        model_statistics['feature'] = feature_name
        model_statistics['model'] = model_name
        model_statistics['split_index'] = int(split_index)
        self.statistics = self.statistics.append(model_statistics, ignore_index=True)

    def add_cv_results(self, model_name, feature_name, cv_results):
        """
        Saves cv statistics for later use in a pipeline attribute

        Args:
            model_name: model name without arguments
            feature_name: feature name without arguments
            cv_results: cv scores for respective feature and model

        """
        if cv_results is None:
            return
        if model_name in self.cv_results:
            if feature_name in self.cv_results[model_name]:
                self.cv_results[model_name][feature_name].append(cv_results)
            else:
                self.cv_results[model_name][feature_name] = [cv_results]
        else:
            self.cv_results[model_name] = {feature_name: [cv_results]}

    def add_learning_curve_data(self, model_name, feature_name, learning_curve_data):
        """
        Saves learning curve for later use in a pipeline attribute

        Args:
            model_name: model name without arguments
            feature_name: feature name without arguments
            learning_curve_data: learning curve data like train sizes and test scores
            for respective feature and model

        """
        if learning_curve_data is None:
            return
        if model_name in self.learning_curve_data:
            if feature_name in self.learning_curve_data[model_name]:
                self.learning_curve_data[model_name][feature_name].append(learning_curve_data)
            else:
                self.learning_curve_data[model_name][feature_name] = [learning_curve_data]
        else:
            self.learning_curve_data[model_name] = {feature_name: [learning_curve_data]}

    def train(self):
        """
        This method starts training for all features, all train test splits and all models
        defined in the config file

        """
        self.mlflow_setup()
        label = self.data_set.get_feature(self.label_name)
        feature_pca = None
        for feature_name in self.feature_names:
            feature = self.data_set.get_feature(feature_name)
            if self.pca_spec and feature_name == self.pca_spec['feature']:
                feature_pca = self.data_set.get_feature_preprocessed_by_pca(feature_name, self.pca_spec['n_components'])
            self.train_with_feature(feature_name, feature, label, feature_pca)

    def train_with_feature(self, feature_name, feature, label, feature_pca=None):
        """
        This method starts training for a specific feature, all train test splits and all models
        defined in the config file

        Args:
            feature_name: name of the feature
            feature: all data of the feature
            label: all data of the label
            feature_pca: all data of the feature preprocessed by pca

        """
        x_train_pca = x_test_pca = None
        for split_index in range(self.train_test_splits):
            train_index, test_index = self.get_next_train_test_indices(feature, label)
            x_train, x_test = feature[train_index], feature[test_index]
            y_train, y_test = label[train_index], label[test_index]
            if self.pca_spec and feature_name == self.pca_spec['feature']:
                x_train_pca, x_test_pca = feature_pca[train_index], feature_pca[test_index]
            self.train_with_dataset_split(split_index, feature_name, x_train, y_train, x_test, y_test,
                                          x_train_pca, x_test_pca)

    def train_with_dataset_split(self, split_index, feature_name, x_train, y_train, x_test, y_test,
                                 x_train_pca=None, x_test_pca=None):
        """
        This method starts training for a specific feature, specific train test split and all models
        defined in the config file

        Args:
            split_index: index of a train test split
            feature_name: name of the feature
            x_train: train data of the feature
            y_train: train data of the label
            x_test: test data of the feature
            y_test: test data of the label
            x_train_pca: train data of the feature preprocessed by pca
            x_test_pca: test data of the feature preprocessed by pca

        """
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
            self.add_learning_curve_data(model_name, feature_name, model.get_learning_curve_data())

    def run_model(self, model_name, model, x_train, y_train, x_test, y_test, drop_nan=True):
        """
        This method starts training for a specific model and a specific train test split

        Args:
            model_name: name of the model to run
            model: model to run
            x_train: train data of the feature
            y_train: train data of the label
            x_test: test data of the feature
            y_test: test data of the label
            drop_nan: if True then deletes all nan rows in pairs: x_train, y_train and x_test, y_test
            to keep the number of elements the same

        """
        param_grid = self.hyper_param_grid[model_name] if model_name in self.hyper_param_grid else None
        if drop_nan:
            nan_x_train = np.argwhere(np.isnan(x_train)).flatten()[::len(x_train.shape)]
            nan_y_train = np.argwhere(np.isnan(y_train)).flatten()[::len(y_train.shape)]
            nan_train = np.unique(np.append(nan_x_train, nan_y_train))
            nan_x_test = np.argwhere(np.isnan(x_test)).flatten()[::len(x_test.shape)]
            nan_y_test = np.argwhere(np.isnan(y_test)).flatten()[::len(y_test.shape)]
            nan_test = np.unique(np.append(nan_x_test, nan_y_test))
            x_train = np.delete(x_train, nan_train, axis=0)
            y_train = np.delete(y_train, nan_train, axis=0)
            x_test = np.delete(x_test, nan_test, axis=0)
            y_test = np.delete(y_test, nan_test, axis=0)
        model.run(x_train, y_train, x_test, y_test, param_grid, self.cv, self.is_learning_curve)

    def get_statistics(self):
        """

        Returns: Statistics of all models with scores like mae, mse a specific feature and train test split

        """
        return self.statistics

    def get_cv_results(self, model_name, feature_name, split_index):
        """

        Args:
            model_name: model name without arguments
            feature_name: feature name without arguments
            split_index: index of a train test split

        Returns: cv scores for respective model and feature

        """
        return self.cv_results[model_name][feature_name][split_index]

    def print_spec(self):
        """
        Prints config specification

        """
        print(yaml.dump(self.spec))

    def get_model_names(self):
        """

        Returns: names of all models without arguments

        """
        return self.model_names

    def plot_grid_search(self, model_name, feature_name, grid_param_name):
        """
        This method plots grid search scores for a specific model, feature and hyper parameter.
        Makes sense for multiple train test splits and if cv parameter was specified in the config file.

        Args:
            model_name: model name without arguments
            feature_name: feature name without arguments
            grid_param_name: hyper parameter name on the x-axis

        """
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

    def plot_models_performance(self, feature_name, column='test_mae'):
        """
        This method plots performance scores for all models and a specific feature.

        Args:
            feature_name: feature name without arguments
            column: Metric or score of the models

        """
        _, ax = plt.subplots(1, 1)
        for model_name in self.model_names:
            model_mask = self.statistics['model'] == model_name
            values = self.statistics[model_mask][['split_index', column]].to_numpy()
            ax.plot(values[:, 0], values[:, 1], '-o', label=model_name)
        ax.set_title(f"Model performance with feature {feature_name}\n", fontsize=16, fontweight='bold')
        ax.set_xlabel('split_index', fontsize=16)
        ax.set_ylabel(column, fontsize=16)
        ax.legend(loc="best", fontsize=15)
        ax.grid('on')

    def plot_learning_curve(self, model_name, feature_name, split_index):
        """
        This method plots learning curve for a specific model, feature and train-test split.

        Args:
            model_name: model name without arguments
            feature_name: feature name without arguments
            split_index: index of a train test split

        """
        train_sizes, train_scores, test_scores, fit_times, _ = \
            self.learning_curve_data[model_name][feature_name][split_index]
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(f"Learning curves for {model_name} with {feature_name}")
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot learning curve
        axes[0].grid()
        axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std, alpha=0.1,
                             color="r")
        axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1,
                             color="g")
        axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                     label="Training score")
        axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                     label="Cross-validation score")
        axes[0].legend(loc="best")

        # Plot n_samples vs fit_times
        axes[1].grid()
        axes[1].plot(train_sizes, fit_times_mean, 'o-')
        axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                             fit_times_mean + fit_times_std, alpha=0.1)
        axes[1].set_xlabel("Training examples")
        axes[1].set_ylabel("fit_times")
        axes[1].set_title("Scalability of the model")

        # Plot fit_time vs score
        axes[2].grid()
        axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
        axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std, alpha=0.1)
        axes[2].set_xlabel("fit_times")
        axes[2].set_ylabel("Score")
        axes[2].set_title("Performance of the model")
