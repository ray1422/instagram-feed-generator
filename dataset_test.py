from torch.utils.data import DataLoader

from dataset import *

ds, = generate_dataset("/home/ray1422/data/ins_dataset/Influencer_brand_dataset", [], debug=True)

train_loader = DataLoader(ds, batch_size=2, shuffle=True)

pxl, ref, ref_mask, cap, cap_mask = next(iter(train_loader))

a = 0
