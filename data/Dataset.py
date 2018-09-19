from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
import os
import numpy as np
from PIL import Image
from pprint import pprint
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
