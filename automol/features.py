import inspect
import pickle
import numpy
import os
import rdkit.Chem as Chem
import rdkit.Chem.Descriptors as Descriptors

"""
Old Feature Generator
Currently kept here just for code presence purposes
"""


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
            for mol in ds.get_feature('molecules')
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
                                        for mol in ds.get_feature('molecules')])
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
                                                 for mol in ds.get_feature('molecules')])
        }
    }

    def __init__(self):
        # feature_name : sub data_frame of features associated with feature_name, lazy_init
        self.__generatable_features = None
        self.__generated_features = {}

    def get_feature(self, data_set, feature_name):
        # if it's represented in the dataset, return directly
        if feature_name in data_set:
            return data_set[feature_name]

        # if dataset doesn't meet requirements, exception
        if not self.requirements_fulfilled(data_set, feature_name):
            raise Exception('requirements for feature %s not satisfied by data set' % feature_name)

        # return already generated feature
        if feature_name in self.__generated_features:
            return self.__generated_features[feature_name]

        # check if generated features are already cached
        indices = data_set.get_indices()
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

        # use the features transform to generate and cache the feature
        if feature_name not in self.__generated_features:
            self.__generated_features[feature_name] = FeatureGenerator.__featureList[feature_name]['transform'](
                data_set)
        if feature_name not in self.__generated_features and not all(cached):
            self.__generated_features[feature_name] = \
                FeatureGenerator.__featureList[feature_name]['transform'](data_set)

        # use cached features
        if all(cached):
            self.__generated_features[feature_name] = numpy.array([v for v in cached_feature.values()])

        # save features
        if not all(cached):
            os.makedirs(feature_dir)
            to_save = {}
            for i, f in enumerate(self.__generated_features[feature_name]):
                to_save[indices[i]] = f
            pickle.dump(to_save, open(feature_path, 'wb'))

        # return the cached feature
        return self.__generated_features[feature_name]

    def requirements_fulfilled(self, data_set, feature_name):
        return not any(req not in data_set for req in FeatureGenerator.__featureList[feature_name]['requirements'])

    def get_acceptable_features(self, data_set, acceptable_types):
        """
            returns list of feature_names that can be offered and are acceptable
        :rtype: list[str]
        """
        return [k for k in self.generatable_features(data_set) if
                FeatureGenerator.__featureList[k]['iam'] & acceptable_types]

    def generatable_features(self, data_set):
        if self.__generatable_features is None:
            self.__generatable_features = {k for k, v, in FeatureGenerator.__featureList.items() if
                                           self.requirements_fulfilled(data_set, k)}
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
