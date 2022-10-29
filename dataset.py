import copy
import csv
import json
import os
import pickle
import random
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import ViTFeatureExtractor

global_data_path = ""
random.seed(48763)


@dataclass(eq=False, frozen=True)
class DataRow:
    post_id: int
    username: str
    sponsored: bool
    caption: str
    photos: List[str]


def job(row) -> Optional[DataRow]:
    global global_data_path
    post_id, username, sponsored, json_file, img_files = row
    post_id = int(post_id)
    img_files = img_files.replace("'", '"')
    sponsored = sponsored != '0'
    img_files = json.loads(img_files)
    img_files = [f"img_resized/{u}" for u in img_files]
    json_file = f"json/{json_file}"
    with open(f"{global_data_path}/{json_file}") as fp:
        obj = json.load(fp)
        if len(obj["edge_media_to_caption"]["edges"]) < 1:
            return None
        caption = obj["edge_media_to_caption"]["edges"][0]["node"]["text"]
        if len(caption) < 10:
            return None
    return DataRow(post_id, username, sponsored, caption, img_files)


def generator_pkl(_dataset_path, debug=False):
    global global_data_path
    global_data_path = _dataset_path
    data_dict: Dict[str, List[DataRow]] = {}  # key: username value: row
    print(f"generating pickle from {global_data_path}")
    with open(f"{global_data_path}/post_info.txt", newline='') as f:
        reader = csv.reader(f, delimiter="\t")
        rows: List[DataRow]
        with Pool(48) as p:
            if debug:
                rows = list(tqdm(p.imap_unordered(job, list(reader)[:1000]), total=1000))
            else:
                rows = list(tqdm(p.imap_unordered(job, reader), total=1601074))

        for row in rows:
            if row is None:
                continue
            if row.username not in data_dict:
                data_dict[row.username] = []
            data_dict[row.username].append(row)

    for k in list(data_dict.keys()):
        if len(data_dict[k]) < 20:
            print(f"delete {k} coz too few posts")
            del data_dict[k]

    pkl_name = "data.pickle" if not debug else "data_debug.pickle"
    with open(f"{_dataset_path}/{pkl_name}", "wb") as f:
        pickle.dump(data_dict, f)
    print(f"{_dataset_path}/{pkl_name} has been written.")
    return data_dict


def generate_dataset(dataset_path, p: List[float], debug=False) -> List['InstagramFeedDataset']:
    """
    note: must import `DataRow` from this file
    :param dataset_path:
    :param p:
    :param debug:
    :return:
    """
    pkl_name = "data.pickle" if not debug else "data_debug.pickle"
    with open(f"{dataset_path}/{pkl_name}", "rb") as f:
        data_dict = pickle.load(f)
    keys = list(data_dict.keys())
    random.shuffle(keys)
    p = copy.deepcopy(p) + [0.]
    for i, v in enumerate(p):
        p[i] += p[i - 1]
    slice_points: List[Optional[int]] = [0] + [int(len(keys) * x) for x in p]
    slice_points[-1] = None

    rets = []
    for i in range(len(slice_points) - 1):
        part_dict = {k: data_dict[k] for k in keys[slice_points[i]:slice_points[i + 1]]}
        rets.append(InstagramFeedDataset(part_dict, dataset_path))

    return rets


class InstagramFeedDataset(Dataset):
    def __init__(self, data_dict: Dict[str, List[DataRow]], dataset_path):
        self.dataset_path = dataset_path
        self.feat_ext = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.data_dict = data_dict

        data_index: List[Tuple[str, int]] = []
        for key, val in tqdm(data_dict.items()):
            # data_index.extend([(key, v) for v in range(len(data_dict[key]))])
            for i, v in enumerate(val):
                pop_val = []
                for file in v.photos:
                    if not os.path.isfile(f"{self.dataset_path}/{file}"):
                        pop_val.append(file)
                for pop_v in pop_val:
                    v.photos.remove(pop_v)
                if len(v.photos) != 0:
                    data_index.append((key, i))

        print("items:", len(data_index))
        random.shuffle(data_index)
        self._data_index = data_index

    def __getitem__(self, index) -> Tuple[np.array, str, str]:
        real_idx = self._data_index[index]
        row = self.data_dict[real_idx[0]][real_idx[1]]
        while True:
            filename = random.sample(row.photos, 1)[0]
            if os.path.isfile(self.dataset_path + "/" + filename):
                break
            print(filename, "not found!")
        try:
            image = Image.open(f"{self.dataset_path}/{filename}")
            pixel_val = self.feat_ext(image, return_tensors="pt").pixel_values
            ref_caption = random.sample(self.data_dict[row.username], 1)[0].caption
        except Exception as e:
            print(e)
            print(filename)
            raise e

        return pixel_val, ref_caption, row.caption

    def __len__(self):
        return len(self._data_index)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <dataset path>")
    load_dataset_path = sys.argv[1]
    generator_pkl(load_dataset_path, debug=len(sys.argv) > 2 and sys.argv[2] == "debug")
