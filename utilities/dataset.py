import glob

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from torchvision.datasets import ImageFolder


def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) >= 1:
        return torch.utils.data.dataloader.default_collate(batch)
    else:
        return None


class COCO_Dataset(Dataset):
    def __init__(self):
        self.path = "/tank/kuntai/COCO_Detection/train2017_reorder/"
        self.len = len(glob.glob(self.path + "*.jpg"))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = Image.open(self.path + "%010d.jpg" % idx).convert("RGB")

        w, h = image.size
        if h > w:
            return None
        transform_hq = T.Compose(
            [
                # T.Pad(
                #     (
                #         math.floor((1280 - w) / 2),
                #         math.floor((720 - h) / 2),
                #         math.ceil((1280 - w) / 2),
                #         math.ceil((720 - h) / 2),
                #     ),
                #     fill=(123, 116, 103),
                # ),
                T.Resize((720, 1280)),
                T.ToTensor(),
            ]
        )
        transform_lq = T.Compose(
            [
                # T.Pad(
                #     (
                #         math.floor((1280 - w) / 2),
                #         math.floor((720 - h) / 2),
                #         math.ceil((1280 - w) / 2),
                #         math.ceil((720 - h) / 2),
                #     ),
                #     fill=(123, 116, 103),
                # ),
                T.Resize((144, 256)),
                T.Resize((720, 1280)),
                T.ToTensor(),
            ]
        )

        return {
            "hq": transform_hq(image),
            "lq": transform_lq(image),
            "fid": idx,
            "video_name": "COCO",
        }


class CityScape(Dataset):
    def __init__(self, train=True):

        if train:
            self.pngs = glob.glob(
                "/tank/kuntai/cityscape/leftImg8bit/train/*/*.png"
            )
        else:
            self.pngs = glob.glob(
                "/tank/kuntai/cityscape/leftImg8bit/val/*/*.png"
            )
        self.train = train

    def __len__(self):
        return len(self.pngs)

    def __getitem__(self, idx):
        image = Image.open(self.pngs[idx]).convert("RGB")

        w, h = image.size
        if h > w:
            return None
        transform_hq = T.Compose([T.Resize((720, 1280)), T.ToTensor(),])
        transform_lq = T.Compose(
            [
                # resize to 240p
                T.Resize((360, 480)),
                # and then to 720p
                T.Resize((720, 1280)),
                T.ToTensor(),
            ]
        )

        return {
            "hq": transform_hq(image),
            "lq": transform_lq(image),
            "fid": idx + 50000 * int(self.train),
            "video_name": "CityScape",
        }


class Test(Dataset):
    def __init__(self, root):

        self.root = root
        self.pngs = glob.glob(root + "/*/*.png")

    def __len__(self):
        return len(self.pngs)

    def __getitem__(self, idx):
        image = Image.open(self.pngs[idx]).convert("RGB")

        w, h = image.size
        if h > w:
            return None
        transform_hq = T.Compose([T.Resize((720, 1280)), T.ToTensor(),])
        transform_lq = T.Compose(
            [
                # resize to 240p
                T.Resize((360, 480)),
                # and then to 720p
                T.Resize((720, 1280)),
                T.ToTensor(),
            ]
        )

        return {
            "hq": transform_hq(image),
            "lq": transform_lq(image),
            "fid": idx,
            "video_name": "Test",
        }


def get_testset(root):

    return Test(root)
