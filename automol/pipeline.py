from __future__ import annotations
import yaml
import pandas as pd
import mlflow
from mlflow_utils.load_env import export_env
from sklearn.model_selection import train_test_split
from automol.datasets import Dataset
from automol.models import generate_all_possible_models
from sklearn.model_selection import ShuffleSplit


class Pipeline:

    def __init__(self, input_yaml_file: str):
        with open(input_yaml_file, 'r') as file:
            try:
                self.spec = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise e
        self.data_set = Dataset.from_spec(self.spec)
        self.models = []
        self.statistics = pd.DataFrame(columns=['model', 'feature', 'training_mae', 'training_mse', 'training_r2_score',
                                                'test_mae', 'test_mse', 'test_r2_score'])
        self.pca_preprocessing = 'pca_preprocessing' in self.spec
        self.pca_spec = {}
        if self.pca_preprocessing:
            self.pca_spec = self.spec['pca_preprocessing']

    def train(self):
        export_env()
        mlflow.set_experiment(self.spec['mlflow_experiment'])
        feature_names = self.spec['features']
        label_name = self.spec['label']
        label = self.data_set.get_feature(label_name)
        self.models = generate_all_possible_models(
            self.spec['problem'], self.spec['models_filter'])
        dataset_split_test_size = self.spec['dataset_split_test_size']
        x_train_pca = x_test_pca = None
        sh_split = ShuffleSplit(n_splits=1, test_size=dataset_split_test_size)
        for feature_name in feature_names:
            feature = self.data_set.get_feature(feature_name)
            train_index, test_index = next(sh_split.split(feature, label))
            x_train, x_test = feature[train_index], feature[test_index]
            y_train, y_test = label[train_index], label[test_index]
            if self.pca_preprocessing and feature_name == self.pca_spec['feature']:
                feature_pca = self.data_set.get_feature_preprocessed_by_pca(feature_name, self.pca_spec['n_components'])
                x_train_pca, x_test_pca = feature_pca[train_index], feature_pca[test_index]
            for model in self.models:
                model_name = str(model)
                if self.pca_preprocessing and feature_name == self.pca_spec['feature'] \
                        and model_name == self.pca_spec['model_name']:
                    n_components = self.pca_spec['n_components']
                    print(f'Running model {model_name} with feature {feature_name} (PCA to {n_components} dimensions).')
                    model.run(x_train_pca, y_train, x_test_pca, y_test)
                else:
                    print(f'Running model {model_name} with feature {feature_name}.')
                    model.run(x_train, y_train, x_test, y_test)
                model_statistics = model.get_statistics()
                model_statistics['feature'] = feature_name
                model_statistics['model'] = model_name
                self.statistics = self.statistics.append(model_statistics, ignore_index=True)

    def get_statistics(self):
        return self.statistics

    def print_spec(self):
        print(yaml.dump(self.spec))
