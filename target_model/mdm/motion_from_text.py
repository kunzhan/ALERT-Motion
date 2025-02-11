import argparse
import os,sys
# from PIL import Image
# from min_dalle import MinDalle
sys.path.append("target_model/mdm")
from sample.generates import Generate
import torch


def get_models(device):
    cfg = None
    model = Generate(device)
    return model, cfg


def generate_motion(model, cfg, text, output_dir, device, num_timesteps, render):
    # model = Generate(device, num_timesteps)

    motion = model.generate_motion(
        text, output_dir, num_timesteps, render
    )
    # save_motion(motion, motion_path)

def mdm_gen_motion_from_text(model, cfg, ori_sent, ori_motion_path, device, num_timesteps=1000, render=False):
    
    generate_motion(model, cfg, ori_sent, ori_motion_path, device, num_timesteps, render)
    
    return


if __name__ == '__main__':
    ori_sent = "The person walked forward and is picking up his toolbox."
    ori_motion_path = "target_model/mdm/target.npy"
    device = 6
    num_timesteps = 1000
    
    model, cfg = get_models(device)
    mdm_gen_motion_from_text(model, cfg, ori_sent, ori_motion_path, device, num_timesteps)