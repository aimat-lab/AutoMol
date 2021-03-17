from __future__ import annotations
from typing import List, TypeVar
import numpy as np
import pandas as pd

from automol.features.feature_generators import FingerprintFeatureGenerator, MoleculeFeatureGenerator, \
    RDkitFeatureGenerator, CoulombMatricesFeatureGenerator, FeatureGenerator

FeatureGeneratorType = TypeVar("FeatureGeneratorType", bound=FeatureGenerator)

_known_feature_generators = [FingerprintFeatureGenerator,
                             MoleculeFeatureGenerator, RDkitFeatureGenerator,
                             CoulombMatricesFeatureGenerator]


def calculate_possible_feature_generators(current_feature_names: List[str],
                                          current_known_feature_generators,
                                          current_feature_generators):
    """
    This method calculates recursively all possible feature generators.

    Args:
        current_feature_names: list of features names which can be used to generate new features,
            increases with recursive steps if successful
        current_known_feature_generators: list of all features generators to try on the current features,
            reduces with recursive steps if successful
        current_feature_generators: list of the current possible features generators,
            increases with recursive steps if successful

    Returns: list of all possible features generators in the ascending order of necessary features,
        i.e. if feature_generator_1 needs feature_A and the feature_generator_2 needs feature_A and feature_B,
        then following list will be returned: [feature_generator_1, feature_generator_2]

    """
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
        """
        Initializes Features with given DataFrame
        features of which are used to calculate all possible feature generators

        Args:
            data: non-empty DataFrame
        """
        self.data = data
        self.__possible_feature_generators__ = calculate_possible_feature_generators(
            self.get_dataset_feature_names(), _known_feature_generators, [])

    def get_dataset_feature_names(self) -> List[str]:
        """

        Returns: feature names of the DataFrame

        """
        return self.data.columns.tolist()

    def check_requested_feature(self, feature_name: str) -> bool:
        """
        Check if a feature can be generated

        Args:
            feature_name: name of the desired feature

        Returns: True if the respective feature generator is in the list

        """
        match = False
        for feature_generator in self.__possible_feature_generators__:
            if feature_generator.generator_data.feature_name == feature_name:
                match = True
        return match

    def generate_feature(self, feature_name: str):
        """
        This method generates not only the requested feature, but also all the necessary features for it,
        and adds these to the current features. No error handling if the feature can not be generated.

        Args:
            feature_name: name of the desired feature

        """
        for feature_generator in self.__possible_feature_generators__:
            fg_feature_name = feature_generator.generator_data.feature_name
            if fg_feature_name not in self.data:
                self.data[fg_feature_name] = feature_generator.transform(self.data).tolist()
                self.data[fg_feature_name] = self.data[fg_feature_name].apply(np.array)
            if fg_feature_name == feature_name:
                break

    def get_feature(self, feature_name: str):
        """
        Getter method to get a feature data.

        Args:
            feature_name: name of the desired feature

        Returns: feature data if the feature is already in the current DataFrame or can be generated, else None

        """
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
