
import os
import re

# 定义文件夹路径
# folder_path = "/path/to/your/folder"

def read_txt(folder_path):

    # 存储数字的列表
    numbers = []
    file_names = []

    # 存储句子的列表
    tar_sentences = []
    sentences = []

    # 遍历文件夹中的txt文件
    for filename in os.listdir(folder_path):
        if filename=='007193.txt':
            continue
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            file_name_without_extension = os.path.splitext(filename)[0]
            file_names.append(file_name_without_extension)
            
            # 打开文件并逐行读取
            with open(file_path, 'r') as file:
                target_found = False
                for line in file:
                    if "target phrase: " in line:
                        # 如果找到目标短语后，将内容添加到句子列表中
                        tar_sentences.append(re.sub(r'target phrase: ', '', line.strip()))


            with open(file_path, 'r') as file:
                lines = file.readlines()
                
                # 寻找文件中最后一个 "new best adv:"
                last_match = None
                for line in reversed(lines):
                    match = re.search(r'new best adv: ([\d.]+)', line)
                    if match:
                        last_match = match
                        break
                
                if last_match:
                    number = float(last_match.group(1))
                    sentence = line.strip()
                    numbers.append(number)
                    sentences.append(re.sub(r'new best adv: ([\d.]+) ', '', sentence))

    return file_names, numbers, sentences, tar_sentences

# txt1, list1, sentence1, tar_sentence1 = read_txt("attack/mdm_run_log_2023-11-04_11-50-56")
# txt2, list2, sentence2, tar_sentence2 = read_txt("attack_copy/run_log_2023-11-05_17-10-50")
# txt3, list3, sentence3, tar_sentence3 = read_txt("attack_MacPromp/mdm_run_log_2023-11-07_20-48-31")
txt1, list1, sentence1, tar_sentence1 = read_txt("attack/run_log_2024-01-04_21-47-37/1/")
txt2, list2, sentence2, tar_sentence2 = read_txt("attack_copy/mld_run_log_2023-12-22_15-49-04/1/")
txt3, list3, sentence3, tar_sentence3 = read_txt("attack_MacPromp/mld_run_log_2023-11-14_20-46-18/1/")

import numpy as np


# diff = np.array(list1)-np.array(list2)
# sorted_indices = np.argsort(np.array(list2))
# sorted_indices = sorted_indices[::-1]
elite_list = txt1#["004601","005935","005946","006058","006186","008900","010546"]
sorted_indices = [i for i, x in enumerate(txt1) if x in elite_list]
sentence = {"target": [], "baseline": [], "llm": [], "mac_prompt": []}
elite = 20
for i in range(elite):
    sentence["target"].append(tar_sentence1[sorted_indices[i]])
    sentence["baseline"].append(sentence1[sorted_indices[i]])
    sentence["llm"].append(sentence2[sorted_indices[i]])
    sentence["mac_prompt"].append(sentence3[sorted_indices[i]])

# import sys
# sys.path.append('target_model/mld')
# from motion_from_texts import get_models, mdm_gen_motion_from_text

# device = 7
# model, cfg = get_models(device)
# key_list = sentence.keys()


# for x in sentence.key():
#     folder_name = f"target_model/mld/mld_to_mdm/mdm/{x}"
#     if not os.path.exists(folder_name):
#         os.makedirs(folder_name)
#     for i in range(elite):
#         id_str = txt1[sorted_indices[i]]
#         gt_motion_path = f"target_model/mld/datasets/HumanML3D/new_joint_vecs/{id_str}.npy"
#         length = (np.load(gt_motion_path)).shape[0]
#         tar_sent = sentence[x][i]
#         tar_motion_path = f"target_model/mld/mld_to_mdm/mdm/{x}/{id_str}.npy"
#         mdm_gen_motion_from_text(model, cfg, tar_sent, tar_motion_path, device, length, render=True)
#         motion = np.transpose(np.load(tar_motion_path), (3, 1, 2, 0)).squeeze(3)
#         np.save(tar_motion_path, motion)

import sys
sys.path.append('target_model/mdm')
from motion_from_text import get_models, mdm_gen_motion_from_text

device = 7
model, cfg = get_models(device)
key_list = sentence.keys()


for x in ['baseline', 'mac_prompt', 'llm']:#'llm', 'target', 
    folder_name = f"mld_to_mdm/{x}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for i in range(elite):
        id_str = txt1[sorted_indices[i]]
        gt_motion_path = f"target_model/mld/datasets/HumanML3D/new_joint_vecs/{id_str}.npy"
        length = 1000#(np.load(gt_motion_path)).shape[0]
        tar_sent = sentence[x][i]
        tar_motion_path = f"mld_to_mdm/{x}/{id_str}.npy"
        mdm_gen_motion_from_text(model, cfg, tar_sent, tar_motion_path, device, length)
        # motion = np.transpose(np.load(tar_motion_path), (3, 1, 2, 0)).squeeze(3) # mdm时需要
        # np.save(tar_motion_path, motion) # mdm时需要
