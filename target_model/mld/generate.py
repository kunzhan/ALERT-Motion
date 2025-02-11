import warnings
warnings.filterwarnings('ignore')

import logging
import os
import time
from builtins import ValueError
# from multiprocessing.sharedctypes import Value
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import ConcatDataset, DataLoader
# from torchsummary import summary
from tqdm import tqdm

from mld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.data.sampling import subsample, upsample
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
import sys
sys.path.append("target_model/mdm")
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion

def generate(model, cfg, text, output_dir, device, length, render, dataset_name='humanml'):
    """
    get input text
    ToDo skip if user input text in command
    current tasks:
         1 text 2 mtion
         2 motion transfer
         3 random sampling
         4 reconstruction

    ToDo 
    1 use one funtion for all expoert
    2 fitting smpl and export fbx in this file
    3 

    """
    # default lengths
    # length = 150
    length = [int(length)]
    text = [text]
    # output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir)

    # mld_time = time.time()

    # sample
    with torch.no_grad():
        rep_lst = []    
        rep_ref_lst = []
        texts_lst = []
        # task: input or Example
        # prepare batch data
        batch = {"length": length, "text": text}
        
        for rep in range(cfg.DEMO.REPLICATION):
            # text motion transfer
            if cfg.DEMO.MOTION_TRANSFER:
                joints = model.forward_motion_style_transfer(batch)
            # text to motion synthesis
            else:
                joints = model(batch, save_path=output_dir)

            # cal inference time
            # infer_time = time.time() - mld_time
            num_batch = 1
            num_all_frame = sum(batch["length"])
            num_ave_frame = sum(batch["length"]) / len(batch["length"])

            # upscaling to compare with other methods
            # joints = upsample(joints, cfg.DATASET.KIT.FRAME_RATE, cfg.DEMO.FRAME_RATE)
            nsample = len(joints)
            id = 0
            for i in range(nsample):
                npypath = str(output_dir)
                with open(npypath.replace(".npy", ".txt"), "w") as text_file:
                    text_file.write(batch["text"][i])
                if render:
                    np.save(npypath, joints[i].detach().cpu().numpy())
                # logger.info(f"Motions are generated here:\n{npypath}")
                
                
    # from mld.data.humanml.utils.plot_script import plot_3d_motion
    skeleton = paramUtil.kit_kinematic_chain if dataset_name == 'kit' else paramUtil.t2m_kinematic_chain
    fig_path = Path(str(npypath).replace(".npy",".mp4"))
    plot_3d_motion(fig_path, skeleton, (joints[0]).numpy(), dataset=dataset_name, title=text[0], fps=25)


# if __name__ == "__main__":
#     text = "the person walked forward and is picking up his toolbox."
#     output_dir = "attack/target.npy"
#     device = 7

#     generate(model, text, output_dir, device)