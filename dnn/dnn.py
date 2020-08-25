
from abc import ABC, abstractmethod

class DNN(ABC):

    # place the model to cpu
    @abstractmethod
    def cpu(self):
        pass

    # place the model to gpu
    @abstractmethod
    def cuda(self):
        pass

    @abstractmethod
    def inference(self, video, requires_grad):
        pass

    # get the name of the class
    def name(self):
        return type(self).__name__