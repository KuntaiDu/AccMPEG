from abc import ABC, abstractmethod


class DNN(ABC):
    # @abstractmethod
    # def cpu(self):
    #     pass

    # @abstractmethod
    # def cuda(self):
    #     pass

    @abstractmethod
    def inference(self, video, requires_grad):
        pass
