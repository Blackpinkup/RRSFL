import sys, os
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_path)
import csv
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import SimpleITK as sitk
import random
import cv2
import torch

class Camelyon17(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        assert split in ['train', 'test']
        assert int(site) in [1,2,3,4,5] # five hospital

        base_path = base_path if base_path is not None else '../data/camelyon17'
        self.base_path = base_path

        data_dict = np.load('../data/camelyon17/data.pkl', allow_pickle=True)
        self.paths, self.labels = data_dict[f'hospital{site}'][f'{split}']

        self.transform = transform
        self.labels = self.labels.astype(np.int64).squeeze()

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, idx):
        img_path = os.path.join(self.base_path, self.paths[idx])
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, label


class DGDR(Dataset):
    """Diabetic retinopathy grading client wrapper.

    Supported layouts:
    - ImageFolder: data/DGDR/{SITE}/{split}/{class_name}/*.jpg
    - CSV: data/DGDR/{SITE}/{split}.csv, data/DGDR/{SITE}-{split}.csv, or
      data/DGDR/{SITE}_{split}.csv with image/path/file and label/grade columns.
    """

    sites = {"APTOS", "DEEPDR", "FGADR", "IDRID", "MESSIDOR", "RLDR"}

    def __init__(self, site, base_path=None, split="train", transform=None):
        site = site.upper()
        if site not in self.sites:
            raise ValueError(f"Unsupported DGDR site: {site}")
        if split not in ["train", "val", "test"]:
            raise ValueError(f"Unsupported DGDR split: {split}")

        self.site = site
        self.split = split
        self.base_path = base_path if base_path is not None else "../data/DGDR"
        self.transform = transform
        self.samples = self._load_samples()
        if len(self.samples) == 0:
            raise FileNotFoundError(f"No DGDR samples found for site={site}, split={split}.")

    def _load_samples(self):
        csv_candidates = [
            os.path.join(self.base_path, self.site, f"{self.split}.csv"),
            os.path.join(self.base_path, f"{self.site}-{self.split}.csv"),
            os.path.join(self.base_path, f"{self.site}_{self.split}.csv"),
        ]
        for csv_path in csv_candidates:
            if os.path.exists(csv_path):
                return self._load_csv(csv_path)

        folder_root = os.path.join(self.base_path, self.site, self.split)
        if os.path.isdir(folder_root):
            return self._load_image_folder(folder_root)

        raise FileNotFoundError(
            "DGDR expects either CSV metadata or an ImageFolder layout under {}".format(
                os.path.join(self.base_path, self.site)
            )
        )

    def _load_csv(self, csv_path):
        samples = []
        with open(csv_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            path_key = next((key for key in ["image", "path", "img", "file", "filename"] if key in columns), None)
            label_key = next((key for key in ["label", "grade", "level", "class"] if key in columns), None)
            if path_key is None or label_key is None:
                raise ValueError(f"{csv_path} must contain an image/path column and a label/grade column.")

            for row in reader:
                img_path = row[path_key]
                if not os.path.isabs(img_path):
                    img_path = os.path.join(os.path.dirname(csv_path), img_path)
                    if not os.path.exists(img_path):
                        img_path = os.path.join(self.base_path, self.site, row[path_key])
                samples.append((img_path, int(row[label_key])))
        return samples

    def _load_image_folder(self, folder_root):
        samples = []
        class_names = sorted(
            [name for name in os.listdir(folder_root) if os.path.isdir(os.path.join(folder_root, name))]
        )
        for class_idx, class_name in enumerate(class_names):
            try:
                label = int(class_name)
            except ValueError:
                label = class_idx
            class_dir = os.path.join(folder_root, class_name)
            for root, _, files in os.walk(class_dir):
                for file_name in files:
                    if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                        samples.append((os.path.join(root, file_name), label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def convert_from_nii_to_png(img):
    high = np.quantile(img,0.99)
    low = np.min(img)
    img = np.where(img > high, high, img)
    lungwin = np.array([low * 1., high * 1.])
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])  
    newimg = (newimg * 255).astype(np.uint8)
    return newimg


class Prostate(Dataset):
    def __init__(self, site, base_path=None, split='train', transform=None):
        channels = {'BIDMC':3, 'HK':3, 'I2CVB':3, 'ISBI':3, 'ISBI_1.5':3, 'UCL':3}
        assert site in list(channels.keys())
        self.split = split
        
        base_path = base_path if base_path is not None else '../data/prostate'
        
        images, labels = [], []
        sitedir = os.path.join(base_path, site)

        ossitedir = np.load("../data/prostate/{}-dir.npy".format(site)).tolist()

        for sample in ossitedir:
            sampledir = os.path.join(sitedir, sample)
            if os.path.getsize(sampledir) < 1024 * 1024 and sampledir.endswith("nii.gz"):
                imgdir = os.path.join(sitedir, sample[:6] + ".nii.gz")
                label_v = sitk.ReadImage(sampledir)
                image_v = sitk.ReadImage(imgdir)
                label_v = sitk.GetArrayFromImage(label_v)
                label_v[label_v > 1] = 1
                image_v = sitk.GetArrayFromImage(image_v)
                image_v = convert_from_nii_to_png(image_v)

                for i in range(1, label_v.shape[0] - 1):
                    label = np.array(label_v[i, :, :])
                    if (np.all(label == 0)):
                        continue
                    image = np.array(image_v[i-1:i+2, :, :])
                    image = np.transpose(image,(1,2,0))
                    
                    labels.append(label)
                    images.append(image)
        labels = np.array(labels).astype(int)
        images = np.array(images)

        index = np.load("../data/prostate/{}-index.npy".format(site)).tolist()

        labels = labels[index]
        images = images[index]

        trainlen = 0.8 * len(labels) * 0.8
        vallen = 0.8 * len(labels) - trainlen
        if(split=='train'):
            self.images, self.labels = images[:int(trainlen)], labels[:int(trainlen)]

        elif(split=='val'):
            self.images, self.labels = images[int(trainlen):int(trainlen + vallen)], labels[int(trainlen):int(trainlen + vallen)]
        else:
            self.images, self.labels = images[int(trainlen + vallen):], labels[int(trainlen + vallen):]

        self.transform = transform
        self.channels = channels[site]
        self.labels = self.labels.astype(np.int64).squeeze()

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform is not None:
            if self.split == 'train':
                R1 = RandomRotate90()
                image, label = R1(image, label)
                R2 = RandomFlip()
                image, label = R2(image, label)

            image = np.transpose(image,(2, 0, 1))
            image = torch.Tensor(image)
            
            label = self.transform(label)

        return image, label


class RandomRotate90:
    def __init__(self, prob=1.0):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            factor = random.randint(0, 4)
            img = np.rot90(img, factor)
            if mask is not None:
                mask = np.rot90(mask, factor)
        return img.copy(), mask.copy()

class RandomFlip:
    def __init__(self, prob=0.75):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)

        return  img, mask

if __name__=='__main__':
    exit()


