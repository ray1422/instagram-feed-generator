import os
from multiprocessing import Pool
import multiprocessing

from PIL import Image
from tqdm import tqdm
from transformers import ViTFeatureExtractor

DATASET_PATH = os.getenv("DATASET_PATH", "/home/ray1422/data/ins_dataset/Influencer_brand_dataset")
N_CPU = multiprocessing.cpu_count()
img_dir = f"{DATASET_PATH}/img_resized"
feat_ext = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")


def job(file):
    try:
        image = Image.open(img_dir + "/" + file)
        pixel_val = feat_ext(image, return_tensors="pt").pixel_values
    except:
        print("invalid image", img_dir + "/" + file)
        os.remove(img_dir + "/" + file)


with Pool(N_CPU) as p:
    img_dir_it = os.listdir(img_dir)
    list(tqdm(p.imap_unordered(job, img_dir_it), total=len(img_dir_it)))
