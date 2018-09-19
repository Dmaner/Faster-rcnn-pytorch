from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
import torch
from pprint import pprint
import argparse
from torchvision import transforms
from skimage.transform import resize

parse = argparse.ArgumentParser()
parse.add_argument('--caffe_pretrain', type=bool, default=False)
opt = parse.parse_args()

example_for_xml = 'E:/my_python/dataset/VOC2012/Annotations/2008_001659.xml'
data_path = 'E:/my_python/dataset/VOC2012'

VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')

def read_image(img_path, dtype=np.float32, color=True):
    f = Image.open(img_path)
    if color:
        img = f.convert('RGB')
    else:
        img = f.convert('P')
    img = np.asarray(img, dtype=dtype)

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))

def voc_xml_read(data_path, img_id, use_diffcult, label_name):
    anno = ET.parse(os.path.join(data_path, 'Annotations', img_id+'.xml'))
    bbox = []
    label = []
    difficult = []
    for obj in anno.findall('object'):
        # if you don't want use difficult object
        if not use_diffcult and int(obj.find('difficult').text) == 1:
            continue
        difficult.append(int(obj.find('difficult').text))
        bndbox_anno = obj.find('bndbox')
        # subtract 1 to make pixel indexes 0-based
        bbox.append([
            int(bndbox_anno.find(tag).text) - 1
            for tag in ('ymin', 'xmin', 'ymax', 'xmax')])
        name = obj.find('name').text.lower().strip()
        label.append(label_name.index(name))
    bbox = np.stack(bbox).astype(np.float32)
    label = np.stack(label).astype(np.int32)
    # When `use_difficult==False`, all elements in `difficult` are False.
    difficult = np.array(difficult, dtype=np.bool).astype(np.uint8)  # PyTorch don't support np.bool

    # Load a image
    img_file = os.path.join(data_path, 'JPEGImages', img_id + '.jpg')
    img = read_image(img_file, color=True)

    # if self.return_difficult:
    #     return img, bbox, label, difficult
    return img, bbox, label, difficult

def bbox_resize(bbox, in_size, out_size):
    bbox = bbox.copy()
    y_scale = float(out_size[0]) / in_size[0]
    x_scale = float(out_size[1]) / in_size[1]
    bbox[:, 0] = y_scale * bbox[:, 0]
    bbox[:, 2] = y_scale * bbox[:, 2]
    bbox[:, 1] = x_scale * bbox[:, 1]
    bbox[:, 3] = x_scale * bbox[:, 3]
    return bbox

def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()

def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img

def preprocess(img, opt, min_size=600, max_size=1000):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = resize(img, (C, H * scale, W * scale), mode='reflect', anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)

class Voc_dataset(Dataset):

    def __init__(self, data_dir, split='trainval',
                 use_difficult=False, return_difficult=False):
        super(Dataset,self).__init__()

        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [id_.strip() for id_ in open(id_list_file)]
        self.data_dir = data_dir
        self.use_difficult = use_difficult
        self.return_difficult = return_difficult
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __getitem__(self, item):
        image_id = self.ids[item]
        img, bbox, label, difficult = voc_xml_read(self.data_dir,
                                                   image_id,
                                                   self.use_difficult,
                                                   self.label_names)
        return img, bbox, label, difficult

    def __len__(self):
        return len(self.ids)

class My_dataset(Dataset):
    def __init__(self, datadir, opt):
        self.voc = Voc_dataset(datadir)
        self.opt = opt

    def __getitem__(self, item):
        ori_img, bbox, label, difficult = self.voc[item]

        _, H, W = ori_img.shape
        img = preprocess(ori_img, self.opt.min_size, self.opt.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = bbox_resize(bbox, (H, W), (o_H, o_W))
        # TODO: Using padding image instead of resize
        return img.copy(), bbox.copy(), label.copy(), scale
    
    def __len__(self):
        return len(self.voc)
