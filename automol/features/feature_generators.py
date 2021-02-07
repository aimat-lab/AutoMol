import inspect
from abc import abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors


@dataclass
class FeatureGeneratorData:
    feature_name: str
    feature_type: str
    requirements: List[str]


class FeatureGenerator:
    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...

    @classmethod
    def create_feature_generator(cls) -> 'FeatureGenerator':

        return cls.__init__(cls)

    @classmethod
    def matches_feature_generator_name(cls, feature_generator_name: str) -> bool:
        return cls.__name__ == feature_generator_name


class MoleculeFeatureGenerator(FeatureGenerator):

    def __init__(self):
        self.feature_generator_data = FeatureGeneratorData(feature_name='molecule', feature_type='abstarct',
                                                           requirements=['smiles'])

    def __call__(self, data_set) -> np.array:
        transform_result = np.array([rdMolDescriptors.CalcCoulombMat(mol)
                                     for mol in data_set.feature_generator().get_feature('molecules')])
        return transform_result


class FingerprintFeatureGenerator(FeatureGenerator):

    def __init__(self):
        self.feature_generator_data = FeatureGeneratorData(feature_name='fingerprint', feature_type='vector',
                                                           requirements=['smiles'])

    def __call__(self, data_set) -> np.array:
        transform_result = np.array([np.array(Chem.RDKFingerprint(mol)).astype(float)
                                     for mol in data_set.feature_generator().get_feature('molecules')])
        return transform_result


class RDkitFeatureGenerator(FeatureGenerator):

    def __init__(self):
        self.feature_generator_data = FeatureGeneratorData(feature_name='rdkit', feature_type='vector',
                                                           requirements=['smiles'])

    def __call__(self, data_set, to_exclude: list = None, sanitize: bool = True, axis: int = 0) -> np.array:
        to_exclude = to_exclude or []
        to_exclude.append('setupAUTOCorrDescriptors')
        calc_props = {k: v for k, v in inspect.getmembers(Descriptors, inspect.isfunction)
                      if not k.startswith('_') and k not in to_exclude}

        transform_result = np.array([[v(mol) for k, v in calc_props.items()] for mol in
                                     data_set.feature_generator().get_feature('molecules')])
        # if sanitize:
        #     df.dropna(axis=axis, how='any', inplace=True)
        return transform_result


class CoulombMatricesFeatureGenerator(FeatureGenerator):

    def __init__(self):
        self.feature_generator_data = FeatureGeneratorData(feature_name='coulomb_matrices', feature_type='vector',
                                                           requirements=['smiles'])

    def __call__(self, data_set) -> np.array:
        transform_result = np.array([np.array(Chem.RDKFingerprint(mol)).astype(float)
                                     for mol in data_set.feature_generator().get_feature('molecules')])
        return transform_result


class CustomFeatureGenerator(FeatureGenerator):

    def __init__(self):
        self.feature_generator_data = FeatureGeneratorData(feature_name='coulomb_matrices', feature_type='vector',
                                                           requirements=['smiles'])

    def __call__(self, data_set) -> np.array:
        pass