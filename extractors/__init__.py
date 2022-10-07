from extractors.superpoint.model import SuperPoint
from extractors.sift.model import SIFT

methods = {
    'SuperPoint': SuperPoint,
    'SIFT': SIFT,
}


def get_feature_extractor(model_name):
    """
    Create method form configuration
    """
    if model_name not in methods:
        raise NameError('{} module was not found among local descriptors. Please choose one of the following '
                        'methods: {}'.format(model_name, ', '.join(methods.keys())))

    return methods[model_name]