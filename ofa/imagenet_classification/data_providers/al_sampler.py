import warnings
import os
import math
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .base_provider import DataProvider
from ofa.utils.my_dataloader import MyRandomResizedCrop, MyDistributedSampler
pwd = os.getcwd()
DATASET_ROOT = "./dataset"
os.makedirs(DATASET_ROOT, exist_ok=True)
print(os.path.abspath(DATASET_ROOT))
MNIST_ROOT = DATASET_ROOT
FMNIST_ROOT = DATASET_ROOT
KMNIST_ROOT = DATASET_ROOT
CALTECH_ROOT = DATASET_ROOT
CIFAR100_ROOT = os.path.join(DATASET_ROOT, "cifar100")
CIFAR10_ROOT = os.path.join(DATASET_ROOT, "cifar10")
SVHN_ROOT = os.path.join(DATASET_ROOT, "SVHN")


__all__ = ["ImagenetALDataProvider", "CIFAR10ALDataProvider", "CIFAR100ALDataProvider", "SubsetSequentialSampler",
           "MNISTALDataProvider", "KMNISTALDataProvider", "FMNISTALDataProvider", "SVHNALDataProvider",
           "CALTECH101ALDataProvider"]

from typing import Sequence
from torch.utils.data.sampler import Sampler


class SubsetSequentialSampler(Sampler[int]):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Args:
        indices (sequence): a sequence of indices
        generator (Generator): Generator used in sampling.
    """
    indices: Sequence[int]

    def __init__(self, indices: Sequence[int]) -> None:
        self.indices = indices

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class ImagenetALDataProvider(DataProvider):
    DEFAULT_PATH = os.getcwd()

    def __init__(
        self,
        save_path=None,
        train_batch_size=256,
        test_batch_size=512,
        valid_size=None,    # init. lab. set has the same size of valid. set
        n_worker=32,
        resize_scale=0.08,
        distort_color=None,
        image_size=224,
        num_replicas=None,
        rank=None,
        lab_idx=None,
        unlab_idx=None,
    ):
        print("This is the AL data loader, the valid_size serves as the ini. lab. set size.")
        warnings.filterwarnings("ignore")
        self._save_path = save_path

        self.image_size = image_size  # int or list of int
        self.distort_color = "None" if distort_color is None else distort_color
        self.resize_scale = resize_scale

        self._valid_transform_dict = {}
        if not isinstance(self.image_size, int):
            from ofa.utils.my_dataloader import MyDataLoader

            assert isinstance(self.image_size, list)
            self.image_size.sort()  # e.g., 160 -> 224
            MyRandomResizedCrop.IMAGE_SIZE_LIST = self.image_size.copy()
            MyRandomResizedCrop.ACTIVE_SIZE = max(self.image_size)

            for img_size in self.image_size:
                self._valid_transform_dict[img_size] = self.build_valid_transform(
                    img_size
                )
            self.active_img_size = max(self.image_size)  # active resolution for test
            valid_transforms = self._valid_transform_dict[self.active_img_size]
            train_loader_class = MyDataLoader  # randomly sample image size for each batch of training image
        else:
            self.active_img_size = self.image_size
            valid_transforms = self.build_valid_transform()
            train_loader_class = torch.utils.data.DataLoader

        train_dataset = self.train_dataset(self.build_train_transform())

        if valid_size is not None:      # serve as the initially labeled set size
            if not isinstance(valid_size, int):
                assert isinstance(valid_size, float) and 0 < valid_size < 1
                valid_size = int(len(train_dataset) * valid_size)

            if lab_idx is None and unlab_idx is None:
                # valid_dataset = self.train_dataset(valid_transforms)
                unlab_indexes, lab_indexes = self.random_sample_valid_set(
                    len(train_dataset), valid_size
                )
                self.lab_indexes = lab_indexes
                self.unlab_indexes = unlab_indexes
            else:
                assert set.union(set(lab_idx), set(unlab_idx)) == set(range(len(train_dataset)))
                unlab_indexes = unlab_idx
                lab_indexes = lab_idx
                self.lab_indexes = lab_indexes
                self.unlab_indexes = unlab_indexes

            lab_sampler = SubsetSequentialSampler(
                lab_indexes
            )
            unlab_sampler = SubsetSequentialSampler(
                unlab_indexes
            )

            self.train = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                sampler=lab_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
            self.unlab = train_loader_class(
                train_dataset,
                batch_size=train_batch_size,
                sampler=unlab_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
            self.valid = None
        else:
            raise ValueError("valid_size (serve as ini lab set size) is None")

        test_dataset = self.test_dataset(valid_transforms)
        if num_replicas is not None:
            test_sampler = torch.utils.data.distributed.DistributedSampler(
                test_dataset, num_replicas, rank
            )
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                sampler=test_sampler,
                num_workers=n_worker,
                pin_memory=True,
            )
        else:
            self.test = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                shuffle=True,
                num_workers=n_worker,
                pin_memory=True,
            )

        if self.valid is None:
            self.valid = self.test


    @staticmethod
    def name():
        return "imagenet"

    @property
    def data_shape(self):
        return 3, self.active_img_size, self.active_img_size  # C, H, W

    @property
    def n_classes(self):
        return 200

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = self.DEFAULT_PATH
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser("~/dataset/imagenet")
        return self._save_path

    @property
    def data_url(self):
        raise ValueError("unable to download %s" % self.name())

    def train_dataset(self, _transforms):
        return datasets.ImageFolder(self.train_path, _transforms)

    def test_dataset(self, _transforms):
        return datasets.ImageFolder(self.valid_path, _transforms)

    @property
    def train_path(self):
        return os.path.join(self.save_path, "train")

    @property
    def valid_path(self):
        return os.path.join(self.save_path, "val")

    @property
    def normalize(self):
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def build_train_transform(self, image_size=None, print_log=True):
        if image_size is None:
            image_size = self.image_size
        if print_log:
            print(
                "Color jitter: %s, resize_scale: %s, img_size: %s"
                % (self.distort_color, self.resize_scale, image_size)
            )

        if isinstance(image_size, list):
            resize_transform_class = MyRandomResizedCrop
            print(
                "Use MyRandomResizedCrop: %s, \t %s"
                % MyRandomResizedCrop.get_candidate_image_size(),
                "sync=%s, continuous=%s"
                % (
                    MyRandomResizedCrop.SYNC_DISTRIBUTED,
                    MyRandomResizedCrop.CONTINUOUS,
                ),
            )
        else:
            resize_transform_class = transforms.RandomResizedCrop

        # random_resize_crop -> random_horizontal_flip
        train_transforms = [
            resize_transform_class(image_size, scale=(self.resize_scale, 1.0)),
            transforms.RandomHorizontalFlip(),
        ]

        # color augmentation (optional)
        color_transform = None
        if self.distort_color == "torch":
            color_transform = transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        elif self.distort_color == "tf":
            color_transform = transforms.ColorJitter(
                brightness=32.0 / 255.0, saturation=0.5
            )
        if color_transform is not None:
            train_transforms.append(color_transform)

        train_transforms += [
            transforms.ToTensor(),
            self.normalize,
        ]

        train_transforms = transforms.Compose(train_transforms)
        return train_transforms

    def build_valid_transform(self, image_size=None):
        if image_size is None:
            image_size = self.active_img_size
        return transforms.Compose(
            [
                transforms.Resize(int(math.ceil(image_size / 0.875))),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def assign_active_img_size(self, new_img_size):
        self.active_img_size = new_img_size
        if self.active_img_size not in self._valid_transform_dict:
            self._valid_transform_dict[
                self.active_img_size
            ] = self.build_valid_transform()
        # change the transform of the valid and test set
        self.valid.dataset.transform = self._valid_transform_dict[self.active_img_size]
        self.test.dataset.transform = self._valid_transform_dict[self.active_img_size]

    def build_sub_train_loader(
        self, n_images, batch_size, num_worker=None, num_replicas=None, rank=None
    ):
        # used for resetting BN running statistics
        if self.__dict__.get("sub_train_%d" % self.active_img_size, None) is None:
            if num_worker is None:
                num_worker = self.train.num_workers

            n_samples = len(self.train.dataset)
            g = torch.Generator()
            g.manual_seed(DataProvider.SUB_SEED)
            rand_indexes = torch.randperm(n_samples, generator=g).tolist()

            new_train_dataset = self.train_dataset(
                self.build_train_transform(
                    image_size=self.active_img_size, print_log=False
                )
            )
            chosen_indexes = rand_indexes[:n_images]
            if num_replicas is not None:
                sub_sampler = MyDistributedSampler(
                    new_train_dataset,
                    num_replicas,
                    rank,
                    True,
                    np.array(chosen_indexes),
                )
            else:
                sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                    chosen_indexes
                )
            sub_data_loader = torch.utils.data.DataLoader(
                new_train_dataset,
                batch_size=batch_size,
                sampler=sub_sampler,
                num_workers=num_worker,
                pin_memory=True,
            )
            self.__dict__["sub_train_%d" % self.active_img_size] = []
            for images, labels in sub_data_loader:
                self.__dict__["sub_train_%d" % self.active_img_size].append(
                    (images, labels)
                )
        return self.__dict__["sub_train_%d" % self.active_img_size]


class CIFAR10ALDataProvider(ImagenetALDataProvider):

    @staticmethod
    def name():
        return "cifar10"

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = CIFAR10_ROOT
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser(f"~/dataset/{self.name()}")
        return self._save_path

    def train_dataset(self, _transforms):
        return datasets.CIFAR10(root=CIFAR10_ROOT, transform=_transforms, train=True, download=True)

    def test_dataset(self, _transforms):
        return datasets.CIFAR10(root=CIFAR10_ROOT, transform=_transforms, train=False, download=True)

    @property
    def train_path(self):
        return self.save_path

    @property
    def valid_path(self):
        return self.save_path


class CIFAR100ALDataProvider(ImagenetALDataProvider):
    @staticmethod
    def name():
        return "cifar100"

    @property
    def n_classes(self):
        return 100

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = CIFAR100_ROOT
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser(f"~/dataset/{self.name()}")
        return self._save_path

    def train_dataset(self, _transforms):
        return datasets.CIFAR100(root=CIFAR100_ROOT, transform=_transforms, train=True, download=True)

    def test_dataset(self, _transforms):
        return datasets.CIFAR100(root=CIFAR100_ROOT, transform=_transforms, train=False, download=True)

    @property
    def train_path(self):
        return self.save_path

    @property
    def valid_path(self):
        return self.save_path


class MNISTALDataProvider(ImagenetALDataProvider):
    @staticmethod
    def name():
        return "mnist"

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = MNIST_ROOT
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser(f"~/dataset/{self.name()}")
        return self._save_path

    def train_dataset(self, _transforms):
        return datasets.MNIST(root=MNIST_ROOT, transform=_transforms, train=True, download=True)

    def test_dataset(self, _transforms):
        return datasets.MNIST(root=MNIST_ROOT, transform=_transforms, train=False, download=True)


class FMNISTALDataProvider(ImagenetALDataProvider):
    @staticmethod
    def name():
        return "fmnist"

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = FMNIST_ROOT
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser(f"~/dataset/{self.name()}")
        return self._save_path

    def train_dataset(self, _transforms):
        return datasets.FashionMNIST(root=FMNIST_ROOT, transform=_transforms, train=True, download=True)

    def test_dataset(self, _transforms):
        return datasets.FashionMNIST(root=FMNIST_ROOT, transform=_transforms, train=False, download=True)


class KMNISTALDataProvider(ImagenetALDataProvider):
    @staticmethod
    def name():
        return "kmnist"

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = KMNIST_ROOT
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser(f"~/dataset/{self.name()}")
        return self._save_path

    def train_dataset(self, _transforms):
        return datasets.KMNIST(root=KMNIST_ROOT, transform=_transforms, train=True, download=True)

    def test_dataset(self, _transforms):
        return datasets.KMNIST(root=KMNIST_ROOT, transform=_transforms, train=False, download=True)


class SVHNALDataProvider(ImagenetALDataProvider):
    @staticmethod
    def name():
        return "svhn"

    @property
    def n_classes(self):
        return 10

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = SVHN_ROOT
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser(f"~/dataset/{self.name()}")
        return self._save_path

    def train_dataset(self, _transforms):
        return datasets.SVHN(root=SVHN_ROOT, transform=_transforms, split="train", download=True)

    def test_dataset(self, _transforms):
        return datasets.SVHN(root=SVHN_ROOT, transform=_transforms, split="test", download=True)


class CALTECH101ALDataProvider(ImagenetALDataProvider):
    @staticmethod
    def name():
        return "caltech101"

    @property
    def n_classes(self):
        return 101

    @property
    def save_path(self):
        if self._save_path is None:
            self._save_path = CALTECH_ROOT
            if not os.path.exists(self._save_path):
                self._save_path = os.path.expanduser(f"~/dataset/{self.name()}")
        return self._save_path

    def train_dataset(self, _transforms):
        if '_train_idx' not in self.__dict__.keys() or '_test_idx' not in self.__dict__.keys():
            self._load_dataset()
        ori_dataset = datasets.Caltech101(root=CALTECH_ROOT, transform=_transforms, download=True)
        ori_dataset.index = [ori_dataset.index[i] for i in self._train_idx]
        ori_dataset.y = [ori_dataset.y[i] for i in self._train_idx]
        return ori_dataset

    def test_dataset(self, _transforms):
        if '_train_idx' not in self.__dict__.keys() or '_test_idx' not in self.__dict__.keys():
            self._load_dataset()
        ori_dataset = datasets.Caltech101(root=CALTECH_ROOT, transform=_transforms, download=True)
        ori_dataset.index = [ori_dataset.index[i] for i in self._test_idx]
        ori_dataset.y = [ori_dataset.y[i] for i in self._test_idx]
        return ori_dataset

    def _load_dataset(self):
        ori = datasets.Caltech101(root=CALTECH_ROOT, download=True)
        dataset_size = len(ori.index)
        dataset_idx = np.arange(dataset_size)
        np.random.seed(0)
        np.random.shuffle(dataset_idx)
        train_size = 0.7
        self._train_idx = dataset_idx[:round(train_size*dataset_size)]
        self._test_idx = dataset_idx[round(train_size*dataset_size):]
