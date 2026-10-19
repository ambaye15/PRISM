__all__ = ["nearest", "predict_parameters", "train_decoder"]


def nearest(*args, **kwargs):
    from .query import nearest as implementation
    return implementation(*args, **kwargs)


def predict_parameters(*args, **kwargs):
    from .model import predict_parameters as implementation
    return implementation(*args, **kwargs)


def train_decoder(*args, **kwargs):
    from .model import train_decoder as implementation
    return implementation(*args, **kwargs)
