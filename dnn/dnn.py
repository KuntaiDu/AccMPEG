from abc import ABC, abstractmethod

from detectron2.data import MetadataCatalog
from PIL import Image
from detectron2.utils.visualizer import Visualizer


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

    @abstractmethod
    def filter_result(self, result, args, gt):
        pass

    # Directly reuse coco normalization since the results are all 
    def visualize(self, image, result, args):
        result = self.filter_result(result, args)
        v = Visualizer(
            image, MetadataCatalog.get('coco_2017_train'), scale=1
        )
        out = v.draw_instance_predictions(result["instances"])
        return Image.fromarray(out.get_image(), "RGB")