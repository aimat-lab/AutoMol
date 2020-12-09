from .datasets import Dataset
import numpy
import pandas as pd
import inspect
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors
from collections import OrderedDict


class FeatureGenerator:

    __featureList = {
        'fingerprint': [
            ['smiles'],
            lambda df:[Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) for smi in df['smiles']]
        ],
    }

    def __init__(self, data_set: Dataset):
        self.data_set = data_set
        self.__generated_features = {}

    def get_feature(self, feature_name):
        if feature_name not in FeatureGenerator.__featureList:
            raise Exception('unknown feature %s' % feature_name)

        if any(req not in self.data_set for req in FeatureGenerator.__featureList[feature_name][0]):
            raise Exception('requirements for feature %s not satisfied by data set' % feature_name)

        if feature_name not in self.__generated_features:
            self.__generated_features[feature_name] = numpy.array(
                FeatureGenerator.__featureList[feature_name][1](self.data_set.data))

        return self.__generated_features[feature_name]


class RdkitDescriptors:

    requires = ['smiles']

    def produce(self, data: pd.DataFrame,
                to_calculate: list = None,
                sanitize: bool = True,
                axis: int = 0) -> pd.DataFrame:
        smiles = data[self.requires[0]].values
        to_calculate = to_calculate or []
        calc_props = OrderedDict(inspect.getmembers(Descriptors, inspect.isfunction))
        for key in list(calc_props.keys()):
            if key.startswith('_'):
                del calc_props[key]
                continue
            if key == 'setupAUTOCorrDescriptors':
                del calc_props[key]
                continue
            if len(to_calculate) != 0 and key not in to_calculate:
                del calc_props[key]

        df = pd.DataFrame(columns=list(calc_props.keys()), index=range(len(smiles)))
        for i, s in enumerate(smiles):
            mol = Chem.MolFromSmiles(s)
            features = [val(mol) for key, val in calc_props.items()]
            df.loc[i, :] = features

        if sanitize:
            df.dropna(axis=axis, how='any', inplace=True)
        return df


"""
class Fingerprint:

    @staticmethod
    def requires():
        return ['smiles']

    @staticmethod
    def produce(data):
        return Chem.RDKFingerprint(Chem.MolFromSmiles(data['smiles']))
"""
