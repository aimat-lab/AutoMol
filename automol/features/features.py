import logging
import os
import pickle
from typing import List, Any, Optional, Dict

import numpy as np
import pandas as pd
import requests

from automol.features.feature_generators import FingerprintFeatureGenerator, MoleculeFeatureGenerator, \
    RDkitFeatureGenerator, CoulombMatricesFeatureGenerator, CustomFeatureGenerator, FeatureGenerator

_known_feature_generators: List[Any] = [FingerprintFeatureGenerator, MoleculeFeatureGenerator, RDkitFeatureGenerator,
                                        CoulombMatricesFeatureGenerator]

logger = logging.getLogger(__name__)


class Features:

    def __init__(self, data_set: pd.DataFrame, custom_features: Optional[Dict]):
        self.possible_feature_generators = self.get_possible_feature_generators()
        if custom_features:
            self.parse_custom_features(custom_features=custom_features)
        self.data_set = data_set
        self.data = data_set.data
        self.generated_features = dict()

    def get_possible_feature_generators(self) -> List[FeatureGenerator]:
        possible_feature_generators = list()
        for feature_generator in _known_feature_generators:
            feature_generator_object = feature_generator.create_feature_generator()
            if feature_generator_object.generator_data.requirements in self.data:
                possible_feature_generators.append(feature_generator_object)
                logger.info(f'Created Feature Generator {feature_generator.__name__} for dataset'
                            f'{self.data_set.__name}')
        return possible_feature_generators

    def parse_custom_features(self, custom_features):
        parsed_features = dict()
        for feature_name, feature_content in custom_features.items():
            file_link = feature_content["file_link"]
            try:
                response = requests.get(file_link)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.info(f"Could not pare custom feature at link {file_link}")
                raise SystemExit(e)
            response_text = response.text
            global_custom_namespace = {}
            exec(response_text, global_custom_namespace)

            custom_feature_generator = CustomFeatureGenerator(feature_name=feature_name,
                                                              feature_type=feature_content['type'],
                                                              requirements=feature_content['requirements'])
            custom_feature_generator.transform = global_custom_namespace[feature_content['function_name']]
            if custom_feature_generator.generator_data.requirements in self.data:
                self.possible_feature_generators.append(custom_feature_generator)
            else:
                logger.info(f"Custom feature {feature_name} is not compatible with the data_class"
                            f"{self.data_set.__name__}")
        return parsed_features

    def check_requested_feature(self, feature_name: str) -> bool:
        match = False
        for feature_generator in self.possible_feature_generators:
            if feature_generator.generator_data.feature_name == feature_name:
                match = True
        return match

    def get_features_from_type(self, feature_type: str):
        feature_generators_from_type = list()
        for feature_generator in self.possible_feature_generators:
            if feature_generator.generator_data.feature_type == feature_type:
                feature_generators_from_type.append(feature_generator)
        return feature_generators_from_type

    def get_feature(self, feature_type: Optional[str], feature_name: Optional[str]):
        # ToDo add option to load features from disk
        # check if requested feature is supported from the dataset
        if feature_name:
            if not self.check_requested_feature(feature_name=feature_name):
                logger.info(f"Requested feature {feature_name} is not compatible for the dataset"
                            f"{self.data_set.__name__}. Skip this feature.")
                return None
        else:
            featur_generators_from_type = self.get_features_from_type(feature_type=feature_type)
        # if it's represented in the dataset, return directly
        if feature_name in self.data:
            return self.data[feature_name]

        # return already generated feature
        if feature_name in self.generated_features:
            return self.generated_features[feature_name]

        # check if generated features are already cached
        indices = self.data.get_indices()
        cached = np.zeros(len(indices), dtype=bool)
        data_set_location = 'data/dsgdb9nsd'
        feature_dir = os.path.join(data_set_location, feature_name)
        feature_path = os.path.join(feature_dir, feature_name + '.p')
        cached_feature = {}
        if os.path.exists(feature_path):
            cached_feature = pickle.load(open(feature_path, "rb"))
        for i, ind in enumerate(indices):
            if ind in cached_feature:
                cached[i] = True

        # use the feature generator to generate and cache the feature
        if feature_name not in self.generated_features and not all(cached):
            pass

        # use cached features
        if all(cached):
            self.generated_features[feature_name] = np.array([v for v in cached_feature.values()])

        # save features
        if not all(cached):
            os.makedirs(feature_dir)
            to_save = {}
            for i, f in enumerate(self.generated_features[feature_name]):
                to_save[indices[i]] = f
            pickle.dump(to_save, open(feature_path, 'wb'))

        # return the cached feature
        return self.generated_features[feature_name]
