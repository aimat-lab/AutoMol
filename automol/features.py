import inspect
import numpy
import pandas

import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors


class Rdkit:

    @staticmethod
    def produce(ds,
                to_exclude: list = None,
                sanitize: bool = True,
                axis: int = 0):
        to_exclude = to_exclude or []
        to_exclude.append('setupAUTOCorrDescriptors')
        calc_props = {k: v for k, v in inspect.getmembers(Descriptors, inspect.isfunction)
                      if not k.startswith('_') and k not in to_exclude}

        return numpy.array([
                [
                    v(mol) for k, v in calc_props.items()
                ]
                for mol in ds.feature_generator().get_feature('molecules')
            ])
        # if sanitize:
        #     df.dropna(axis=axis, how='any', inplace=True)


class FeatureGenerator:
    __featureList = {
        'molecules': {
            'iam': {'abstract'},
            'requirements': ['smiles'],
            'transform': lambda ds: numpy.array([Chem.MolFromSmiles(smi) for smi in ds.data['smiles']])
        },
        'fingerprint': {
            'iam': {'vector'},
            'requirements': ['smiles'],
            'transform':
                lambda ds: numpy.array([numpy.array(Chem.RDKFingerprint(mol)).astype(float)
                                    for mol in ds.feature_generator().get_feature('molecules')])
        },
        'rdkit': {
            'iam': {'vector'},
            'requirements': ['smiles'],
            'transform': lambda ds: Rdkit.produce(ds)
        },
        'coulomb_mat_eigen': {
            'iam': {'vector'},
            'requirements': ['smiles'],
            'transform': lambda ds: numpy.array([Chem.rdMolDescriptors.CalcCoulombMat(mol)
                                                 for mol in ds.feature_generator().get_feature('molecules')])
        }
    }

    def __init__(self, data_set):
        self.data_set = data_set

        # feature_name : sub data_frame of features associated with feature_name, lazy_init
        self.__generatable_features = None
        self.__generated_features = {}
        self.__custom_featureList = {}

    def add_custom_features(self, custom_features:dict):
        overlap = self.__featureList.keys() & custom_features.keys()
        if overlap:
            raise Exception(
                'feature %s already exists, and can not be added as a new custom feature' % next(iter(overlap)))

        self.__custom_featureList = {**self.__custom_featureList, **custom_features}

    def clear_custom_features(self):
        self.__custom_featureList = {}

    def get_feature(self, feature_name):
        # if it's represented in the dataset, return directly
        if feature_name in self.data_set:
            return self.data_set[feature_name]

        # if dataset doesn't meet requirements, exception
        if not self.requirements_fulfilled(feature_name):
            raise Exception('requirements for feature %s not satisfied by data set' % feature_name)

        # use the features transform to generate and cache the feature
        if feature_name not in self.__generated_features:
            self.__generated_features[feature_name] =\
                FeatureGenerator.__featureList[feature_name]['transform'](self.data_set)

        # return the cached feature
        return self.__generated_features[feature_name]

    def requirements_fulfilled(self, feature_name):
        return not any(req not in self.data_set for req in FeatureGenerator.__featureList[feature_name]['requirements'])

    def get_acceptable_features(self, acceptable_types):
        """
            returns list of feature_names that can be offered and are acceptable
        :rtype: list[str]
        """
        return [k for k in self.generatable_features() if FeatureGenerator.__featureList[k]['iam'] & acceptable_types]

    def generatable_features(self):
        if self.__generatable_features is None:
            self.__generatable_features =\
                {k for k, v, in FeatureGenerator.__featureList.items() if self.requirements_fulfilled(k)}
        return self.__generatable_features


"""
class Fingerprint:

    @staticmethod
    def requires():
        return ['smiles']

    @staticmethod
    def produce(data):
        return Chem.RDKFingerprint(Chem.MolFromSmiles(data['smiles']))
"""
