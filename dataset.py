# Custom dataset
from PIL import Image
import torch.utils.data as data
import os
import random


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, subfolder='train', direction='AtoB', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(DatasetFromFolder, self).__init__()
        self.input_path = os.path.join(image_dir, subfolder)
        self.image_filenames = [x for x in sorted(os.listdir(self.input_path))]
        self.direction = direction
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr

    def __getitem__(self, index):
        # Load Image
        img_fn = os.path.join(self.input_path, self.image_filenames[index])
        img = Image.open(img_fn)
        if self.direction == 'AtoB':
            input = img.crop((0, 0, img.width // 2, img.height))
            target = img.crop((img.width // 2, 0, img.width, img.height))
        elif self.direction == 'BtoA':
            input = img.crop((img.width // 2, 0, img.width, img.height))
            target = img.crop((0, 0, img.width // 2, img.height))

        # preprocessing
        if self.resize_scale:
            input = input.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            target = target.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            input = input.crop((x, y, x + self.crop_size, y + self.crop_size))
            target = target.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.fliplr:
            if random.random() < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            input = self.transform(input)
            target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
