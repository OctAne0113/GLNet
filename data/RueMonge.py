import torch
from PIL import Image
import numpy as np
from utils.base_tools import get_name
import json
from sklearn.model_selection import train_test_split
import cv2

# 新数据集的映射关系和颜色定义
ignore_label = 255
num_classes = 8  # 包含背景，共8个类
colors = {
    0: [0, 0, 0],      # 背景
    1: [250, 170, 30],  # 窗户 (对应原 facade 的 window)
    2: [70, 70, 70],    # 墙壁 (对应原 facade 的 building)
    3: [107, 142, 35],  # 阳台 (对应原 facade 的 tree)
    4: [153, 153, 153],  # 门 (对应原 facade 的 door)
    5: [255, 0, 255],  # 屋檐 (对应原 facade 的 sky)
    6: [70, 130, 180],  # 天空 (对应原 facade 的 sky)
    7: [0, 0, 142]      # 商店 (对应原 facade 的 car)
}

class PolygonTrans():
    def __init__(self):
        # 更新类别名称及对应的编号
        self.binary = {
            "background": 0,
            "window": 1,
            "wall": 2,
            "balcony": 3,
            "door": 4,
            "eaves": 5,
            "sky": 6,
            "shop": 7
        }
        # 覆盖顺序保持不变
        self.overlap_order = ["sky", "eaves", "wall", "door", "window", "balcony", "shop"]

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
        "background": [],
        "window": [],
        "wall": [],
        "balcony": [],
        "door": [],
        "eaves": [],
        "sky": [],
        "shop": []
    }
    record_rectangle = {
        "background": [],
        "window": [],
        "wall": [],
        "balcony": [],
        "door": [],
        "eaves": [],
        "sky": [],
        "shop": []
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
    roots = args.root + "classall/translated_data_runemonge_train/"
    root_test = args.root + "classall/translated_data_ruemonge_test/"

    items = get_name(roots + "images", mode_folder=False)
    items_test = get_name(root_test + "images", mode_folder=False)

    record = []
    for item in items:
        record.append([roots + "images/" + item, roots + "binary_mask/" + item])

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
        # if self.args.use_ignore:
        #     self.id_to_trainid = {7: 255}

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
