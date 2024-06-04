import numpy as np
import os
import shutil
from tqdm import tqdm

total_files = os.listdir("/home/niksrid/bc_algos/dataset_v3_test_16hz")

for i in tqdm(range(len(total_files))):
    file_name = total_files[i]
    if file_name.split(".")[-1] == "gzip":
        ori_path = os.path.join("/home/niksrid/bc_algos/dataset_v3_test_16hz", file_name)
        target_path = os.path.join("/home/niksrid/bc_algos/datasets/dataset_v3_test_16hz", file_name)
        shutil.copy(ori_path, target_path)
