from __future__ import annotations
import yaml
import pandas as pd
import mlflow
from mlflow_utils.load_env import export_env
from sklearn.model_selection import train_test_split
from automol.datasets import Dataset
from automol.models import ModelGenerator


def get_train_test_set(feature, label, train_ratio=0.8):
    return train_test_split(feature, label, test_size = 1 - train_ratio)

def get_train_valid_test_set(feature, label, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2):
    x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size = 1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + val_ratio))
    return (x_train, y_train, x_val, y_val, x_test, y_test)


class Pipeline:

    def __init__(self, input_yaml_file: str):
        with open(input_yaml_file, 'r') as file:
            try:
                self.spec = yaml.safe_load(file)
            except yaml.YAMLError as e:
                raise e
        self.data_set = Dataset.from_spec(self.spec)
        self.model_generator = ModelGenerator()
        self.models = []
        self.statistics = pd.DataFrame(columns=['model', 'feature', 'training_mae', 'training_mse', 'training_r2_score',\
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
        self.models = self.model_generator.generate_all_possible_models(
                self.spec['problem'], self.spec['models_filter'])
        train_ratio, val_ratio, test_ratio = self.spec['train_valid_test_split']
        feature_pca = None
        for feature_name in feature_names:
            feature = self.data_set.get_feature(feature_name)
            for model in self.models:
                model_name = str(model)
                if self.pca_preprocessing and feature_name == self.pca_spec['feature']\
                                          and model_name == self.pca_spec['model_name']:
                    n_components = self.pca_spec['n_components']
                    print(f'Running model {model_name} with feature {feature_name} (PCA to {n_components} dimensions).')
                    feature_pca = self.data_set.get_feature_preprocessed_by_pca(feature_name, n_components)
                    x_train, x_test, y_train, y_test = get_train_test_set(feature_pca, label, train_ratio)
                else:
                    print(f'Running model {model_name} with feature {feature_name}.')
                    x_train, x_test, y_train, y_test = get_train_test_set(feature, label, train_ratio)
                model.run(x_train, y_train, x_test, y_test)
                model_statistics = model.get_statistics()
                model_statistics['feature'] = feature_name
                model_statistics['model'] = model_name
                self.statistics = self.statistics.append(model_statistics, ignore_index=True)

    def get_statistics(self):
        return self.statistics

    def print_spec(self):
        print(yaml.dump(self.spec))
