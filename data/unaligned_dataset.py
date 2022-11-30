import os.path
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np
import matplotlib.pyplot as plt
from options.train_options import TrainOptions
import data.random_pair as random_pair
import torchvision.transforms as transforms
import torch
import matplotlib.pyplot as plt



class TrainDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def initialize(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        #opt.phase
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.dir_Seg = os.path.join(opt.dataroot, 'label')
        self.Atarget_paths = sorted(make_dataset(self.dir_Seg, opt.max_dataset_size))
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B  # get the size of dataset B
        self.input_nc = self.opt.output_nc       # get the number of channels of input image
        self.output_nc = self.opt.input_nc      # get the number of channels of output image
        if not self.opt.isTrain:
            self.skipcrop = True
            self.skiprotate = True
        else:
            self.skipcrop = False
            self.skiprotate = False

        if self.skipcrop:
            osize = [opt.crop_size, opt.crop_size]
        else:
            osize = [opt.load_size, opt.load_size]

        if self.skiprotate:
            angle = 0
        else:
            angle = opt.angle

        transform_list = []
        transform_list.append(random_pair.randomrotate_pair(angle))    # scale the image
        self.transforms_rotate = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))    # scale the image
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Scale(osize, Image.NEAREST))    # scale the segmentation
        self.transforms_seg_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_pair.randomcrop_pair(opt.crop_size))    # random crop image & segmentation
        self.transforms_crop = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        self.transforms_normalize = transforms.Compose(transform_list)


        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        Seg_path = self.Atarget_paths[index % self.A_size]
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        mode = 'L' if (self.input_nc == self.output_nc) else 'RGB'
        A_img = Image.open(A_path).convert(mode)
        Seg_img = Image.open(Seg_path).convert('I')
        B_img = Image.open(B_path).convert(mode)
        A_img = self.transforms_scale(A_img)
        B_img = self.transforms_scale(B_img)
        Seg_img = self.transforms_seg_scale(Seg_img)
        
        if not self.skiprotate:
            [A_img, Seg_img] = self.transforms_rotate([A_img, Seg_img])
            [B_img] = self.transforms_rotate([B_img])

        if not self.skipcrop:
            [A_img, Seg_img] = self.transforms_crop([A_img, Seg_img])
            [B_img] = self.transforms_crop([B_img])
        
        
        A_img = self.transforms_toTensor(A_img)
        B_img = self.transforms_toTensor(B_img)
        Seg_img = self.transforms_toTensor(Seg_img)
        

        A_img = self.transforms_normalize(A_img)
        B_img = self.transforms_normalize(B_img)

        #kesheng 
        Seg_img[Seg_img > 0] = 1

        Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.crop_size, self.opt.crop_size)
        Seg_imgs[0, :, :] = Seg_img == 1

        return {'A': A_img, 'B': B_img, 'Seg': Seg_imgs, 'Seg_one': Seg_img,
                'A_paths': A_path, 'B_paths': B_path, 'Seg_paths': Seg_path}

        
    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
    
    def name(self):
        return 'TrainDataset'

class TestDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')
        self.dir_seg = os.path.join(opt.dataroot, opt.phase + 'target')

        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        self.B_size = len(self.B_paths)
        self.seg_paths = sorted(make_dataset(self.dir_seg, opt.max_dataset_size))

        if not self.opt.isTrain:
            self.skipcrop = True
            self.skiprotate = True
        else:
            self.skipcrop = False
            self.skiprotate = False

        if self.skipcrop:
            osize = [opt.crop_size, opt.crop_size]
        else:
            osize = [opt.load_size, opt.load_size]

        if self.skiprotate:
            angle = 0
        else:
            angle = opt.angle

        transform_list = []
        transform_list.append(random_pair.randomrotate_pair(angle))    # scale the image
        self.transforms_rotate = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))    # scale the image
        self.transforms_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Scale(osize, Image.NEAREST))    # scale the segmentation
        self.transforms_seg_scale = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(random_pair.randomcrop_pair(opt.crop_size))    # random crop image & segmentation
        self.transforms_crop = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.ToTensor())
        self.transforms_toTensor = transforms.Compose(transform_list)

        transform_list = []
        transform_list.append(transforms.Normalize([0.5], [0.5]))
        self.transforms_normalize = transforms.Compose(transform_list)

    def __getitem__(self, index):
        B_path = self.B_paths[index % self.B_size]
        B_img = Image.open(B_path).convert('L')
        B_img = self.transforms_scale(B_img)
        B_img = self.transforms_toTensor(B_img)
        B_img = self.transforms_normalize(B_img)

        Seg_path = self.seg_paths[index % self.B_size]
        Seg_img = Image.open(Seg_path).convert('I')
        Seg_img = self.transforms_scale(Seg_img)
        Seg_img = self.transforms_toTensor(Seg_img)
        
        Seg_img[Seg_img > 0] = 1
        Seg_imgs = torch.Tensor(self.opt.output_nc_seg, self.opt.crop_size, self.opt.crop_size)
        Seg_imgs[0, :, :] = Seg_img == 1


        return {'B': B_img, 'Seg': Seg_img,
                'B_paths': B_path, 'Seg_paths': Seg_path}

    def __len__(self):
        return self.B_size

    def name(self):
        return 'TestDataset'

if __name__ == "__main__":
    opt = TrainOptions().parse()
    data = TrainDataset(opt)
    print(data[1]['B_paths'],data[1]['B_paths'])
    plt.imshow(data[1]['B'][0], cmap='gray')
    plt.show()
    print(data[1]['Be_paths'])
    plt.imshow(data[1]['Be'][0], cmap='gray')
    plt.show()
