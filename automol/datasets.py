import glob
from abc import ABC
from typing import Optional, Dict

from automol.features.features import FeatureGenerator, Features

import os
import pandas as pd
from rdkit import Chem

import pysftp
import paramiko


class Dataset(ABC):

    def __init__(self, data: pd.Dataframe, custom_features: Optional[Dict] = None):
        assert not data.empty, "The data set is empty. Please select the correct path to the data."
        self.data = data
        self.features = Features(custom_features)

    def get_feature(self, feature_type: str, feature_name: Optional[str] = None):
        return self.features.get_feature(feature_name=feature_name, feature_type=feature_type)

    def split(self, split_right):
        return Dataset(self.data[0: split_right]), Dataset(self.data[split_right:])

    @classmethod
    def from_spec(cls, spec):
        class_name = spec['dataset_class']
        class_ = globals().get(class_name)
        if class_ is None or not issubclass(class_, cls):
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
        if 'custom_features' in spec:
            return class_(data, spec['custom_features'])
        else:
            return class_(data)

    @classmethod
    def parse(cls, text) -> dict:
        pass

    def get_indices(self) -> list:
        return self.data.index

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

    def get_indices(self) -> list:
        return ["{:06d}".format(index) for index in self.data['index']]
