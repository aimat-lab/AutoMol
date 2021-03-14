from __future__ import annotations
from typing import List, TypeVar, Optional, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from automol.datasets import Dataset
import logging
import os
import pickle

import numpy
import requests
import pandas as pd

from automol.features.feature_generators import FingerprintFeatureGenerator, MoleculeFeatureGenerator, \
    RDkitFeatureGenerator, CoulombMatricesFeatureGenerator, CustomFeatureGenerator, FeatureGenerator

FeatureGeneratorType = TypeVar("FeatureGeneratorType", bound=FeatureGenerator)

_known_feature_generators: List[FeatureGeneratorType] = [FingerprintFeatureGenerator, MoleculeFeatureGenerator,
                                                         RDkitFeatureGenerator,
                                                         CoulombMatricesFeatureGenerator]

logger = logging.getLogger(__name__)


def calculate_possible_feature_generators(current_feature_names, current_known_feature_generators=[], current_feature_generators=[]):
    #print(f'calculate_possible_feature_generators')
    #print(f'1. current_feature_names: {current_feature_names}')
    #print(f'2. current_known_feature_generators: {current_known_feature_generators}')
    #print(f'3. current_feature_generators: {current_feature_generators}')
    for feature_generator in current_known_feature_generators:
        feature_generator_object = feature_generator.get_instance()
        if any(a in current_feature_names for a in feature_generator_object.generator_data.requirements):
            current_feature_names.append(feature_generator_object.generator_data.feature_name)
            current_known_feature_generators.remove(feature_generator)
            current_feature_generators.append(feature_generator_object)
            return calculate_possible_feature_generators(current_feature_names,
                                                         current_known_feature_generators,
                                                         current_feature_generators)
    return current_feature_generators

class Features:

    def __init__(self, data: pd.DataFrame, custom_features = {}):
        self.data = data
        self.generated_features = []

        self.__possible_feature_generators__ = calculate_possible_feature_generators(
            self.get_dataset_feature_names(), _known_feature_generators)
        print(f'Features possible_feature_generators: {self.__possible_feature_generators__}')
        #self.custom_features = custom_features
        #self.add_custom_features()
        #self.calculate_possible_feature_generators()

    def get_possible_feature_generators(self):
        return self.__possible_feature_generators__

    def get_dataset_feature_names(self):
        return list(self.data.columns)

    def add_custom_features(self):
        if bool(self.custom_features):
            self.custom_features.update(custom_features)
            self.__possible_feature_generators__ = self.calculate_possible_feature_generators()


    def calculate_possible_feature_generators(self) -> List[FeatureGenerator]:
        print(f'Features calculate_possible_feature_generators')
        possible_feature_generators = []
        # adds commonly used known features
        for feature_generator in _known_feature_generators:
            print(f'Features for feature_generator {feature_generator} in _known_feature_generators')
            feature_generator_object = feature_generator.get_instance()
            if any(a in self.data for a in feature_generator_object.generator_data.requirements):
                print(f'Features if any')
                possible_feature_generators.append(feature_generator_object)
                print(f'Features possible_feature_generators.append {feature_generator_object}')
                #logger.info(f'Created Feature Generator {feature_generator.__name__} for dataset')
                            #f'{self.data.__name__}')
                            #f'{self.data}')
        # adds custom features wanted for this features instance
        #if bool(self.custom_features):
            #possible_feature_generators.extend(self.parse_custom_features(self.custom_features))
        return possible_feature_generators

    def get_acceptable_feature_gens(self, acceptable_types):
        return [gen for gen in self.__possible_feature_generators__ if
                set(gen.generator_data.feature_type) & set(acceptable_types)]

    def parse_custom_features(self, custom_features):
        parsed_features = []
        for feature_name, feature_content in custom_features.items():
            file_link = feature_content["file_link"]
            try:
                response = requests.get(file_link)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logger.info(f"Could not parse custom feature at link {file_link}")
                raise SystemExit(e)
            response_text = response.text
            global_custom_namespace = {}
            exec(response_text, global_custom_namespace)

            custom_feature_generator = CustomFeatureGenerator(feature_name=feature_name,
                                                              feature_type=feature_content['type'],
                                                              requirements=feature_content['requirements'])
            custom_feature_generator.transform = global_custom_namespace[feature_content['function_name']]
            if custom_feature_generator.generator_data.requirements in self.data:
                parsed_features.append(custom_feature_generator)
            else:
                logger.info(f"Custom feature {feature_name} is not compatible with the data_class"
                            #f"{self.data_set.__name__}")
                            f'{self.data_set}')
        return parsed_features

    def check_requested_feature(self, feature_name: str) -> bool:
        match = False
        for feature_generator in self.__possible_feature_generators__:
            if feature_generator.generator_data.feature_name == feature_name:
                match = True
        return match

    def get_features_from_type(self, feature_type: str):
        feature_generators_from_type = list()
        for feature_generator in self.__possible_feature_generators__:
            if feature_generator.generator_data.feature_type == feature_type:
                feature_generators_from_type.append(feature_generator)
        return feature_generators_from_type

    def generate_feature(self, feature_name):
        for feature_generator in self.__possible_feature_generators__:
            fg_feature_name = feature_generator.generator_data.feature_name
            if not fg_feature_name in self.data:
                self.data[fg_feature_name] = feature_generator.transform(self.data).tolist()
            if fg_feature_name == feature_name:
                break

    def get_feature(self, feature_name):
        if feature_name in self.get_dataset_feature_names():
            print(f'Got the feature {feature_name} from the current dataset.')
            return self.data[feature_name]
        print(f'Checking if the feature {feature_name} can be generated.')
        if self.check_requested_feature(feature_name):
            self.generate_feature(feature_name)
            print(f'Got the generated feature {feature_name}.')
            return self.data[feature_name]
        else:
            print(f'The feature {feature_name} can not be generated.')
        return None

        # check if requested feature is supported from the dataset
        if feature_name:
            if not self.check_requested_feature(feature_name=feature_name):
                logger.info(f"Requested feature {feature_name} is not compatible for the dataset")
                            #f"{self.data_set.__name__}. Skip this feature.")
                            #f'{self.data_set}. Skip this feature.')
                return None
        # if it's represented in the dataset, return directly
        if feature_name in self.data:
            return self.data[feature_name]

        # return already generated feature
        if feature_name in self.generated_features:
            return self.generated_features[feature_name]

        # check if generated features are already cached
        indices = self.data.get_indices()
        cached = numpy.zeros(len(indices), dtype=bool)
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
            self.generated_features[feature_name] = numpy.array([v for v in cached_feature.values()])

        # save features
        if not all(cached):
            os.makedirs(feature_dir)
            to_save = {}
            for i, f in enumerate(self.generated_features[feature_name]):
                to_save[indices[i]] = f
            pickle.dump(to_save, open(feature_path, 'wb'))

        # return the cached feature
        return self.generated_features[feature_name]
