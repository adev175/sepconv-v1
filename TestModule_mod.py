from PIL import Image
import torch
from torchvision import transforms
from math import log10
from torchvision.utils import save_image as imwrite
from torch.autograd import Variable
import os


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class Vimeo90k:
    #gom gt_dir và input_dir -> data_dir,
    def __init__(self, data_root):
        self.data_root = data_root
        self.image_root = os.path.join(self.data_root, 'sequences')
        test_fn = os.path.join(self.data_root, 'tri_testlist.txt')
        self.transform = transforms.Compose([transforms.ToTensor()])
        with open(test_fn, 'r') as f:
            self.im_list = f.read().splitlines()

        self.input0_list = []
        self.input1_list = []
        self.gt_list = []
        for item in self.im_list:
            imgpath = os.path.join(self.image_root, self.im_list[item])
            self.input0_list.append(to_variable(self.transform(Image.open(imgpath + '/' + '/im1.png')).unsqueeze(0)))
            self.input1_list.append(to_variable(self.transform(Image.open(imgpath + '/' + '/im3.png')).unsqueeze(0)))
            self.gt_list.append(to_variable(self.transform(Image.open(imgpath + '/' +  '/im2.png')).unsqueeze(0)))

    def Test(self, model, output_dir, logfile=None, output_name='output.png'):
        av_psnr = 0
        if logfile is not None:
            logfile.write('{:<7s}{:<3d}'.format('Epoch: ', model.epoch.item()) + '\n')
        for idx in range(len(self.im_list)):
            if not os.path.exists(output_dir + '/' + self.im_list[idx]):
                os.makedirs(output_dir + '/' + self.im_list[idx])
            frame_out = model(self.input0_list[idx], self.input1_list[idx])
            gt = self.gt_list[idx]
            psnr = -10 * log10(torch.mean((gt - frame_out) * (gt - frame_out)).item())
            av_psnr += psnr
            imwrite(frame_out, output_dir + '/' + self.im_list[idx] + '/' + output_name, range=(0, 1))
            msg = '{:<15s}{:<20.16f}'.format(self.im_list[idx] + ': ', psnr) + '\n'
            print(msg, end='')
            if logfile is not None:
                logfile.write(msg)
        av_psnr /= len(self.im_list)
        msg = '{:<15s}{:<20.16f}'.format('Average: ', av_psnr) + '\n'
        print(msg, end='')
        if logfile is not None:
            logfile.write(msg)
