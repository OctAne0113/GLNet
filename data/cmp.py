import torch
from PIL import Image
import numpy as np
import json
from sklearn.model_selection import train_test_split
import cv2
from utils.base_tools import get_name

ignore_label = 255
num_classes = 12
colors = {
    0: [0, 0, 0],       # 背景
    1: [128, 0, 0],     # facade
    2: [0, 128, 0],     # molding
    3: [128, 128, 0],   # cornice
    4: [0, 0, 128],     # pillar
    5: [128, 0, 128],   # window
    6: [0, 128, 128],   # door
    7: [128, 128, 128], # sill
    8: [64, 0, 0],      # blind
    9: [192, 0, 0],     # balcony
    10: [64, 128, 0],   # shop
    11: [192, 128, 0],  # deco
}


class PolygonTrans():
    def __init__(self):
        # 更新类别名称及对应的编号
        self.binary = {
            "facade": 1,
            "molding": 2,
            "cornice": 3,
            "pillar": 4,
            "window": 5,
            "door": 6,
            "sill": 7,
            "blind": 8,
            "balcony": 9,
            "shop": 10,
            "deco": 11,

        }
        # 覆盖顺序
        self.overlap_order = [
             "facade", "molding", "cornice", "pillar",
            "window", "door", "sill", "blind", "balcony", "shop", "deco"
        ]

    def polygon2mask(self, img_size, polygons, rectangles):
        mask = np.zeros(img_size, dtype=np.uint8)
        for cat in self.overlap_order:
            polygon = polygons[cat]
            cv2.fillPoly(mask, polygon, color=self.binary[cat])
            rectangle = rectangles[cat]
            for ret in rectangle:
                x1, y1 = ret[0]
                x2, y2 = ret[1]
                mask[y1:y2, x1:x2] = self.binary[cat]
        return mask

    # translate label_id to color img
    def id2trainId(self, label, select=None):
        w, h = label.shape
        label_copy = np.zeros((w, h, 3), dtype=np.uint8)
        for index, color in colors.items():
            if select is not None:
                if index == select:
                    label_copy[label == index] = color
                else:
                    continue
            else:
                label_copy[label == index] = color
        return label_copy.astype(np.uint8)


def read_json(file_name):
    # 按新类别调整
    record = {
        "facade": [], "molding": [], "cornice": [], "pillar": [],
        "window": [], "door": [], "sill": [], "blind": [],
        "balcony": [], "shop": [], "deco": [],
    }
    record_rectangle = {
        "facade": [], "molding": [], "cornice": [], "pillar": [],
        "window": [], "door": [], "sill": [], "blind": [],
        "balcony": [], "shop": [], "deco": [],
    }

    with open(file_name, "r") as load_polygon:
        data = json.load(load_polygon)

    data = data["shapes"]
    for item in data:
        label = item["label"]
        points = item["points"]
        shape = item["shape_type"]
        if label not in record:
            continue

        if shape == "rectangle":
            record_rectangle[label].append(np.array(points, dtype=np.int32))
        else:
            record[label].append(np.array(points, dtype=np.int32))

    return record, record_rectangle


def prepare_facade_data(args):
    roots = args.root + "classall/translated_data_cmp_train12/"
    # roots_extra = args.root + "classall/translated_data_c6extra/"
    root_test = args.root + "classall/translated_data_cmp_test12/"

    items = get_name(roots + "images", mode_folder=False)
    # items_extra = get_name(roots_extra + "images", mode_folder=False)
    items_test = get_name(root_test + "images", mode_folder=False)

    record = []
    for item in items:
        record.append([roots + "images/" + item, roots + "binary_mask/" + item])
    # record_extra = []
    # for items_extra1 in items_extra:
    #     record_extra.append([roots_extra + "images/" + items_extra1, roots_extra + "binary_mask/" + items_extra1])
    record_test = []
    for item1 in items_test:
        record_test.append([root_test + "images/" + item1, root_test + "binary_mask/" + item1])

    # 直接划分训练集和验证集
    train, val = train_test_split(record, train_size=0.9, random_state=1)

    test = record_test



    return {"train": train, "val": val, "test": test}


class Facade(torch.utils.data.Dataset):
    def __init__(self, args, mode, joint_transform=None, standard_transform=None):
        self.args = args
        self.imgs = prepare_facade_data(args)[mode]
        if len(self.imgs) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

        self.joint_transform = joint_transform
        self.standard_transform = standard_transform


    def __getitem__(self, index):
        img_path, mask_path = self.imgs[index]
        img, mask = Image.open(img_path).convert('RGB'), Image.open(mask_path)

        mask = np.array(mask)
        mask_copy = mask.copy()
        if self.args.use_ignore:
            for k, v in self.id_to_trainid.items():
                mask_copy[mask == k] = v
        mask = Image.fromarray(mask_copy.astype(np.uint8))

        if self.joint_transform is not None:
            img, mask = self.joint_transform(img, mask)

        if self.standard_transform is not None:
            img = self.standard_transform(img)

        return {"images": img, "masks": torch.from_numpy(np.array(mask, dtype=np.int32)).long()}

    def __len__(self):
        return len(self.imgs)