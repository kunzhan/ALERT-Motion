import numpy as np

import codecs as cs
import json 
def read_ids():
    split_file = "attack_copy/render.txt"# os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list

id_list = read_ids()

for id_str in id_list:

    arr = np.load(f"target_model/mld/MM/MDM/{id_str}.npy")
    # N = arr.shape[-1]   
    reshaped_arr = arr.transpose(3,1,2,0).squeeze(3)
    np.save(f"target_model/mld/MM/MDM/{id_str}.npy", reshaped_arr)

# arr1 = np.load("target_model/mld/MM/MLD/000565.npy")
# arr2 = np.load("target_model/mld/MM/MDM/000565.npy")

# print(np.linalg.norm(arr1), np.linalg.norm(arr2))