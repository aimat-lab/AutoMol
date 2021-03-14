from __future__ import annotations
from typing import List, TYPE_CHECKING

import inspect
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


@dataclass
class GeneratorData:
    feature_name: str
    feature_type: List[str]
    requirements: List[str]

    def __init__(self, feature_name, feature_type, requirements):
        self.feature_name = feature_name
        self.feature_type = feature_type if type(feature_type) is list else [feature_type]
        self.requirements = requirements


class FeatureGenerator:

    def __init__(self, generator_data):
        self.generator_data: GeneratorData = generator_data

    @abstractmethod
    def transform(self, data: pd.DataFrame):
        ...

    # singleton
    @classmethod
    def get_instance(cls) -> 'FeatureGenerator':
        #print(f'FeatureGenerator get_instance cls: {cls}')
        if cls is FeatureGenerator:
            raise Exception("can't initialize abstract feature generator")
        elif cls.__instance__ is None:
            cls.__instance__ = cls()
        #print(cls.__instance__, type(cls.__instance__))
        #print(dir(cls.__instance__))
        return cls.__instance__


class MoleculeFeatureGenerator(FeatureGenerator):

    __instance__: FeatureGenerator = None

    def __init__(self):
        super().__init__(GeneratorData(feature_name='molecules', feature_type='molecules', requirements=["smiles"]))

    def transform(self, data: pd.DataFrame) -> np.array:
        transform_result = np.array([Chem.MolFromSmiles(smi) for smi in data['smiles']])
        return transform_result


class FingerprintFeatureGenerator(FeatureGenerator):

    __instance__: FeatureGenerator = None

    def __init__(self):
        super().__init__(GeneratorData(feature_name='fingerprint', feature_type='vector',
                                       requirements=["molecules"]))

    def transform(self, data: pd.DataFrame) -> np.array:
        transform_result = np.array([np.array(Chem.RDKFingerprint(mol)).astype(float)
                                     for mol in data['molecules']])
        return transform_result


class RDkitFeatureGenerator(FeatureGenerator):

    __instance__: FeatureGenerator = None

    def __init__(self):
        super().__init__(GeneratorData(feature_name='rdkit', feature_type='vector', requirements=["molecules"]))

    def transform(self, data: pd.DataFrame) -> np.array:
        to_exclude = list()
        to_exclude.append('setupAUTOCorrDescriptors')
        calc_props = {k: v for k, v in inspect.getmembers(Descriptors, inspect.isfunction)
                      if not k.startswith('_') and k not in to_exclude}

        transform_result = np.array([[v(mol) for k, v in calc_props.items()] for mol in
                                     data['molecules']])
        # if sanitize:
        #     df.dropna(axis=axis, how='any', inplace=True)
        return transform_result


class CoulombMatricesFeatureGenerator(FeatureGenerator):

    __instance__: FeatureGenerator = None

    def __init__(self):
        super().__init__(GeneratorData(feature_name='coulomb_matrices', feature_type='vector',
                                       requirements=["molecules"]))

    def transform(self, data: pd.DataFrame) -> np.array:
        transform_result = np.array([Chem.rdMolDescriptors.CalcCoulombMat(mol)
                                     for mol in data['molecules']])
        return transform_result


class CustomFeatureGenerator(FeatureGenerator):

    def __init__(self, feature_name: str, feature_type: str, requirements: List[str]):
        super().__init__(GeneratorData(feature_name=feature_name, feature_type=feature_type,
                                       requirements=requirements))

    def transform(self, data_set: pd.DataFrame) -> np.array:
        pass
