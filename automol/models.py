from typing import List, Dict, Optional, Any


class ModelGenerator:

    def __init__(self):
        pass

    def generate_all_possible_models(self,
                                     data_set,
                                     problem_type,
                                     models_filters: Optional[List[Dict[str, Any]]] = None) -> List['Model']:
        r = []

        for models_filter in models_filters:
            if models_filter['git_uri'] == 'sklearn':
                import automol.sklearn.models
                r += automol.sklearn.models.SklearnModelGenerator().generate_all_possible_models(
                    data_set, problem_type, models_filter)
            else:
                raise Exception('unknown model type specified %s' % models_filter['type'])

        return r


class Model:

    def run(self, sets, labels):
        pass
