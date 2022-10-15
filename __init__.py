from models.openglue import *
from models.superglue.model import SuperGlue
from models.imc2022.model_v2 import IMCNet
# from models.imc2022.model import IMCNet

methods = {
    'SuperGlue': SuperGlue,
    'IMCNet': IMCNet,
}


def get_matching_module(model_name):
    if model_name not in methods:
        raise NameError('{} module was not found among matching models. Please choose one of the following '
                        'methods: {}'.format(model_name, ', '.join(methods.keys())))
    
    return methods[model_name]