from __future__ import annotations
import glob
from abc import ABC
from automol.features.features import Features
import os
import pandas as pd
from rdkit import Chem
import pysftp
import paramiko
import numpy as np
from sklearn import decomposition


class Dataset(ABC):

    def __init__(self, data: pd.DataFrame):
        self.data = data
        if self.data.empty:
            raise Exception("dataset empty")
        self.__features = None

    def features(self):
        if self.__features is None:
            self.__features = Features(self.data)
        return self.__features

    def get_feature(self, feature_name: str):
        """
        Wrapper on getting feature from generator

        Args:
            feature_name: name of the desired feature

        """
        feature = self.features().get_feature(feature_name)
        if feature is None:
            return None
        return np.stack(feature)

    def get_feature_preprocessed_by_pca(self, feature_name: str, n_components=50):
        pca = decomposition.PCA(n_components=n_components)
        feature = self.get_feature(feature_name)
        pca.fit(feature)
        return pca.transform(feature)

    @classmethod
    def from_spec(cls, spec):
        class_name = spec['dataset_class']
        class_ = globals().get(class_name)
        if class_ is None or not issubclass(type(class_), type) or not issubclass(class_, cls):
            raise Exception("%s is not a valid class name" % class_name)

        data = {}
        amount = spec.get('amount', -1)

        def parse_and_catch(text):
            nonlocal amount
            try:
                data.update((k, data[k] + [v] if k in data else [v]) for k, v in class_.parse(text).items())
                amount -= 1
            except Exception as e:
                with open("erroneous.txt", "w") as out:
                    out.write(text)
                    out.write('\n' + str(e))
                raise

        # lsdf dataset
        if spec['dataset_location'].startswith('lsdf://'):
            import io
            config = paramiko.config.SSHConfig()
            config.parse(open(os.path.expanduser('~/.ssh/config')))
            conf = config.lookup('lsdf')
            lsdf_dataset_path = spec['dataset_location'][len('lsdf://'):]
            with pysftp.Connection(host=conf['hostname'],
                                   username=conf['user'], private_key=conf['identityfile'][0]) as sftp:
                with sftp.cd(lsdf_dataset_path):
                    for file_attr in sftp.sftp_client.listdir_iter():
                        if amount == 0:
                            break
                        file = file_attr.filename
                        f = io.BytesIO()
                        sftp.getfo(file, f)
                        f.seek(0)
                        text = io.TextIOWrapper(f, encoding='utf-8').read()
                        parse_and_catch(text)
        # local dataset
        else:
            if not os.path.isdir(spec['dataset_location']):
                raise Exception("path %s doesn't exist" % spec['dataset_location'])

            for fn in glob.iglob(spec['dataset_location'] + '/*'):
                if amount == 0:
                    break
                with open(fn) as f:
                    text = f.read()
                parse_and_catch(text)

        data = pd.DataFrame(data, columns=data.keys())

        return class_(data)

    @classmethod
    def parse(cls, text) -> dict:
        pass

    def __iter__(self):
        return self.data.__iter__()

    def __getitem__(self, key):
        return self.data[key]


class QM9(Dataset):

    @classmethod
    def parse(cls, text):
        r = {}
        lines = str.splitlines(text)
        r['atom_count'] = int(lines[0])
        line2 = lines[1].strip().split()
        r['tag'] = line2.pop(0)
        r['index'], r['A'], r['B'], r['C'], r['mu'], r['alpha'], r['homo'], r['lumo'], r['gap'], r['r2'], r['zpve'], r[
            'U0'], r['U'], r['H'], r['G'], r['Cv'], = [float(a) for a in line2]
        r['smiles'] = Chem.MolToSmiles(Chem.MolFromSmiles(lines[3 + r['atom_count']].split('\t')[0], sanitize=True),
                                       isomericSmiles=False, canonical=True)
        r['index'] = int(r['index'])
        return r
