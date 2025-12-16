import os
import glob
from torch.utils.data import Dataset
from PIL import Image

class ImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        # 支持传入单个文件路径；若 root 是文件则直接使用该文件
        if os.path.isfile(root):
            self.filenames = [root]
        else:
            self.filenames = sorted([*glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True),
                                     *glob.glob(os.path.join(root, '**', '*.png'), recursive=True)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)
        
        if self.transforms:
            img = self.transforms(img)
        
        return img
