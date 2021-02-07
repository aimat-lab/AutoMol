import logging
import pickle
from typing import List, Any

import numpy as np
import pandas as pd
import os

from automol.features.feature_generators import FingerprintFeatureGenerator, MoleculeFeatureGenerator, \
    RDkitFeatureGenerator, CoulombMatricesFeatureGenerator, CustomFeatureGenerator, FeatureGenerator

_known_feature_generators: List[Any] = [FingerprintFeatureGenerator, MoleculeFeatureGenerator, RDkitFeatureGenerator,
                                        CoulombMatricesFeatureGenerator, CustomFeatureGenerator]

logger = logging.getLogger(__name__)


class Features:

    def __init__(self, data_set: pd.DataFrame, acceptable_feature_type: str):
        self.data_set = data_set
        self.acceptable_feature_names = self.get_acceptable_feature_names(acceptable_feature_type=
                                                                          acceptable_feature_type)
        self._generated_features = {}

    def requirements_fulfilled(self, feature_name) -> bool:
        req_fulfilled = not any(feature_generator.feature_generator_data.requirements not in self.data_set for
                                feature_generator in _known_feature_generators if
                                feature_generator.feature_generator_data.feature_name == feature_name)
        return req_fulfilled

    def get_acceptable_feature_names(self, acceptable_feature_type: str):
        """
            returns list of feature_names that can be offered and are acceptable
        :rtype: list[str]
        """
        possible_feature_generators = [
            feature_generator for feature_generator in _known_feature_generators if
            self.requirements_fulfilled(feature_generator.FeatureGeneratorData.feature_name)]
        acceptable_feature_names = [feature_generator.feature_generator_data.feature_name for feature_generator in
                                    possible_feature_generators if
                                    feature_generator.feature_generator_data.feature_type == acceptable_feature_type]
        return acceptable_feature_names

    def get_feature(self, feature_name):
        # if it's represented in the dataset, return directly
        if feature_name in self.data_set:
            return self.data_set[feature_name]

        # if dataset doesn't meet requirements, exception
        if not self.requirements_fulfilled(feature_name):
            raise Exception('requirements for feature %s not satisfied by data set' % feature_name)

        # return already generated feature
        if feature_name in self._generated_features:
            return self._generated_features[feature_name]

        # check if generated features are already cached
        indices = self.data_set.get_indices()
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

        # use the features transform to generate and cache the feature
        if feature_name not in self._generated_features and not all(cached):
            self._generated_features[feature_name] = \
                FeatureGenerator.__featureList[feature_name]['transform'](self.data_set)

        # use cached features
        if all(cached):
            self._generated_features[feature_name] = np.array([v for v in cached_feature.values()])

        # save features
        if not all(cached):
            os.makedirs(feature_dir)
            to_save = {}
            for i, f in enumerate(self._generated_features[feature_name]):
                to_save[indices[i]] = f
            pickle.dump(to_save, open(feature_path, 'wb'))

        # return the cached feature
        return self._generated_features[feature_name]

    @staticmethod
    def resolve_metrics(feature_generators_names: List[str]) -> List[FeatureGenerator]:
        feature_generators = list()
        for feature_generators_name in feature_generators_names:
            for feature_generator in _known_feature_generators:
                if feature_generator.matches_feature_generator_name(feature_generators_name=feature_generators_name):
                    feature_generators.append(feature_generator.create_metric())
                    logger.info(f'Created Feature Generator {feature_generators_name}')
        return feature_generators

"""
class Fingerprint:

    @staticmethod
    def requires():
        return ['smiles']

    @staticmethod
    def produce(data):
        return Chem.RDKFingerprint(Chem.MolFromSmiles(data['smiles']))
"""
