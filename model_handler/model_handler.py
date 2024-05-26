import abc


class ModelHandler(metaclass=abc.ABCMeta):
    def __init__(self, device):
        self.device = device
        self.model = self.load_model()
        self.input_size = self.load_input_size()
        self.labels = self.load_labels()

    @abc.abstractmethod
    def load_model(self) -> any:
        pass

    @abc.abstractmethod
    def load_labels(self) -> list[str]:
        pass

    @abc.abstractmethod
    def load_input_size(self) -> int:
        pass

    # TODO: add trial for optuna
    @abc.abstractmethod
    def train(self, dataset, eval_function):
        pass

    @abc.abstractmethod
    def save_model(self, path):
        pass
