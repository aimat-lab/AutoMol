from __future__ import annotations
from typing import List, TypeVar
import numpy as np
import pandas as pd

from automol.features.feature_generators import FingerprintFeatureGenerator, MoleculeFeatureGenerator, \
    RDkitFeatureGenerator, CoulombMatricesFeatureGenerator, FeatureGenerator

FeatureGeneratorType = TypeVar("FeatureGeneratorType", bound=FeatureGenerator)

_known_feature_generators: List[FeatureGeneratorType] = [FingerprintFeatureGenerator, MoleculeFeatureGenerator,
                                                         RDkitFeatureGenerator,
                                                         CoulombMatricesFeatureGenerator]

def calculate_possible_feature_generators(current_feature_names,
                                          current_known_feature_generators=[],
                                          current_feature_generators=[]):
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

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.__possible_feature_generators__ = calculate_possible_feature_generators(
            self.get_dataset_feature_names(), _known_feature_generators)

    def get_dataset_feature_names(self) -> list[str]:
        return self.data.columns.tolist()

    def check_requested_feature(self, feature_name: str) -> bool:
        match = False
        for feature_generator in self.__possible_feature_generators__:
            if feature_generator.generator_data.feature_name == feature_name:
                match = True
        return match

    def generate_feature(self, feature_name: str):
        for feature_generator in self.__possible_feature_generators__:
            fg_feature_name = feature_generator.generator_data.feature_name
            if not fg_feature_name in self.data:
                self.data[fg_feature_name] = feature_generator.transform(self.data).tolist()
                self.data[fg_feature_name] = self.data[fg_feature_name].apply(np.array)
            if fg_feature_name == feature_name:
                break

    def get_feature(self, feature_name: str):
        if feature_name in self.get_dataset_feature_names():
            print(f'Got the feature {feature_name} from the current dataset.')
            return self.data[feature_name]
        print(f'Checking if the feature {feature_name} can be generated.')
        if self.check_requested_feature(feature_name):
            self.generate_feature(feature_name)
            print(f'Got the generated feature {feature_name}.')
            return self.data[feature_name]
        else:
            print(f'The feature {feature_name} could not be generated.')
        return None
