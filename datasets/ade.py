from torch.utils.data import Dataset
import imageio
from . import transforms
from PIL import Image
import torchvision.transforms as T
import os
import numpy as np
import cv2

# ADE20K 150 classes
class_list = [
    'wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag'
]

def load_img_name_list(img_name_list_path):
    img_name_list = np.loadtxt(img_name_list_path, dtype=str)
    return img_name_list

def load_cls_label_list(name_list_dir):
    return np.load(os.path.join(name_list_dir, 'cls_labels_onehot.npy'), allow_pickle=True).item()

class ADE20KDataset(Dataset):
    def __init__(
            self,
            root_dir='/data/DatasetCollection/ADEChallengeData2016',
            name_list_dir='datasets/ade',
            split='train',
            stage='train',
            tasks='100-10',
            step=0,
    ):
        super().__init__()

        self.root_dir = root_dir
        self.stage = stage
        self.tasks = tasks
        
        if stage == 'train':
            self.img_dir = os.path.join(root_dir, 'images/training')
            self.label_dir = os.path.join(root_dir, 'annotations/training')
        else:
            self.img_dir = os.path.join(root_dir, 'images/validation')
            self.label_dir = os.path.join(root_dir, 'annotations/validation')

        self.name_list_dir = os.path.join(name_list_dir, 'incremental_split',
                                          split + '_' + tasks + '_step_' + str(step + 1) + '.txt')
        self.name_list = load_img_name_list(self.name_list_dir)

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        _img_name = self.name_list[idx]
        img_name = os.path.join(self.img_dir, _img_name + '.jpg')
        label_name = os.path.join(self.label_dir, _img_name + '.png')

        image = np.asarray(Image.open(img_name).convert('RGB'))
        label = np.asarray(Image.open(label_name))
        
        # ADE20K labels: 0 (ignore), 1-150 (classes)
        # Shift to: 255 (ignore), 0-149 (classes)
        label = label.astype(np.int16) - 1
        label[label == -1] = 255 

        return _img_name, image, label

class ADE20KSegDataset(ADE20KDataset):
    def __init__(self,
                 root_dir='/data/DatasetCollection/ADEChallengeData2016',
                 name_list_dir='datasets/ade',
                 split='train',
                 stage='train',
                 tasks='100-10',
                 step=0,
                 crop_size=512,
                 ignore_index=255,
                 aug=False,
                 scales=None,
                 **kwargs):

        super().__init__(root_dir, name_list_dir, split, stage, tasks, step)

        self.aug = aug
        self.ignore_index = ignore_index
        self.crop_size = crop_size
        self.scales = scales
        self.color_jittor = transforms.PhotoMetricDistortion()
        self.gaussian_blur = transforms.GaussianBlur(p=0.5)

    def __len__(self):
        return len(self.name_list)

    def __transforms(self, image, label):
        if self.aug:
            # Multi-scale augmentation
            if self.scales:
                image, _, label = transforms.random_scaling(image, None, label, self.scales)

            # random_fliplr returns (image, label, depth, normal)
            image, label, _, _ = transforms.random_fliplr(image, label)
            image = self.color_jittor(image)
            image = Image.fromarray(image.astype(np.uint8))
            image = self.gaussian_blur(image)
            image = np.array(image)
            if self.crop_size:
                # random_crop returns (crop_image, label, img_box)
                image, label, _ = transforms.random_crop(
                    image, label, crop_size=self.crop_size, 
                    mean_rgb=(123.675, 116.28, 103.53), 
                    ignore_index=self.ignore_index
                )
        
        image = transforms.normalize_img(image)
        image = np.transpose(image, (2, 0, 1))
        return image, label

    def __getitem__(self, idx):
        img_name, image, label = super().__getitem__(idx)
        image, label = self.__transforms(image=image, label=label)
        return img_name, image, label

class ADE20KTrainDataset(ADE20KSegDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __getitem__(self, idx):
        return super().__getitem__(idx)
