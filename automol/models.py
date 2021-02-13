from typing import List, Dict, Optional
import numpy

from automol.features.feature_generators import FeatureGenerator


class ModelGenerator:

    def __init__(self):
        pass

    def generate_all_possible_models(self, data_set, problem_type, models_filters: Optional[List[Dict[str]]] = None) \
            -> List['Model']:
        r = []

        for models_filter in models_filters:
            if models_filter['type'] == 'sklearn':
                import automol.sklearn.models
                r.append(automol.sklearn.models.SklearnModelGenerator()
                         .generate_all_possible_models(data_set, problem_type, models_filter))
            else:
                raise Exception('unknown model type specified %s' % models_filter['type'])

        return r


class Model:

    def __init__(self, core, feature_gen: FeatureGenerator):
        self.core = core
        self.feature_gen: FeatureGenerator = feature_gen

    def fit(self, data_set, labels):
        inputs = self.feature_gen.transform(data_set.data)
        labels = numpy.array(data_set[labels]).flatten()
        # print(type(self.core))
        # print(labels.shape, labels)
        self.core.fit(inputs, labels)

    def predict(self, data_set):
        return self.core.predict(self.feature_gen.transform(data_set.data))

    def __str__(self):
        return self.core.__str__().replace("()", "")
