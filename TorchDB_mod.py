import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random


class DBreader_frame_interpolation(Dataset):
    """
    This code sets the data_root and is_training arguments as class attributes,
    reads the train and test file lists from the dataset,
    and creates a transforms pipeline for data augmentation
    that includes random cropping, horizontal and vertical flipping,
    color jittering, and conversion to a tensor.
    With these modifications, the DBreader_frame_interpolation class
    will use the same data augmentation pipeline and dataset source as the VimeoTriplet class.
    """

    def __init__(self, data_root, resize=None, is_training=None):
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        self.training = is_training
        # print('data root:',data_root)
        # print('self.image_root :',self.image_root)
        train_fn = os.path.join(self.data_root, 'tri_trainlist.txt')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        # print("train test root :",train_fn)
        with open(train_fn, 'r') as f:
            self.trainlist = f.read().splitlines()
        with open(test_fn, 'r') as f:
            self.testlist = f.read().splitlines()

        # print('train list:', self.trainlist)
        if resize is not None:
            self.transforms = transforms.Compose([
                transforms.Resize(resize),
                # transforms.RandomHorizontalFlip(0.5),
                # transforms.RandomVerticalFlip(0.5),
                transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor()
            ])


    def __getitem__(self, index):
        if self.training:
            imgpath = os.path.join(self.image_root, self.trainlist[index])
        else:
            imgpath = os.path.join(self.image_root, self.testlist[index])
        imgpaths = [imgpath + '/im1.png', imgpath + '/im2.png', imgpath + '/im3.png']
        # print('img path:', imgpaths)
        # Load images
        img1 = Image.open(imgpaths[0])
        img2 = Image.open(imgpaths[1])
        img3 = Image.open(imgpaths[2])

        # Data augmentation
        seed = random.randint(0, 2 ** 32)
        random.seed(seed)
        img1 = self.transforms(img1)
        random.seed(seed)
        img2 = self.transforms(img2)
        random.seed(seed)
        img3 = self.transforms(img3)
        # Random Temporal Flip
        if random.random() >= 0.5:
            img1, img3 = img3, img1
            imgpaths[0], imgpaths[2] = imgpaths[2], imgpaths[0]
        # else:
        #     T = transforms.ToTensor()
        #     img1 = T(img1)
        #     img2 = T(img2)
        #     img3 = T(img3)

        imgs = [img1, img2, img3]

        return imgs

    def __len__(self):
        if self.training:
            return len(self.trainlist)
        else:
            return len(self.testlist)
        return 0
