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

    def __init__(self, custom_features: Optional[Dict]):
        self.feature_generators = Features.get_feature_generators()
        if custom_features:
            self.parse_custom_features(custom_features=custom_features)
        self.generated_features = dict()

    @staticmethod
    def get_feature_generators() -> List[FeatureGenerator]:
        possible_feature_generators = list()
        for feature_generator in _known_feature_generators:
            possible_feature_generators.append(feature_generator.create_feature_generator())
            logger.info(f'Created Feature Generator {feature_generator.__name__}')
        return possible_feature_generators

    def get_acceptable_feature_generators(self, acceptable_feature_type: str):
        acceptable_feature_generators = [feature_generator for feature_generator in self.possible_feature_generators if
                                         feature_generator.generator_data.feature_type == acceptable_feature_type]
        return acceptable_feature_generators

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
            self.feature_generators.append(custom_feature_generator)
        return parsed_features

    def get_feature(self, data_set: pd.DataFrame, feature_name: str):
        # ToDo add option to load features from disk
        # ToDo check feature requirements dynamically.
        # ToDo think how to better incorporate the feature type info for creating features
        # if it's represented in the dataset, return directly
        if feature_name in data_set:
            return data_set[feature_name]

        # return already generated feature
        if feature_name in self.generated_features:
            return self.generated_features[feature_name]

        # check if generated features are already cached
        indices = data_set.get_indices()
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
