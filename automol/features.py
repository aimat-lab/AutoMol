from .datasets import *


import numpy
from rdkit import Chem


class FeatureGenerator:

    __featureList = {
        'fingerprint': [
            ['smiles'],
            lambda df:[Chem.RDKFingerprint(Chem.MolFromSmiles(smi)) for smi in df['smiles']]
        ],
    }

    def __init__(self, data_set:Dataset):
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


"""
class Fingerprint:

    @staticmethod
    def requires():
        return ['smiles']

    @staticmethod
    def produce(data):
        return Chem.RDKFingerprint(Chem.MolFromSmiles(data['smiles']))
"""
