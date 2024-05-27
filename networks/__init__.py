from networks.model_global_local_adversarial import GlobalLocalAdversarialModel
from networks.model_recognize import RecognizeModel
from networks.model_writer_identify import WriterIdentifyModel

all_models = {
    "gl_adversarial_model": GlobalLocalAdversarialModel,
    "recognize_model": RecognizeModel,
    "identifier_model": WriterIdentifyModel,
}


def get_model(name):
    return all_models[name]
