import argparse
import os,sys
# from PIL import Image
# from min_dalle import MinDalle
sys.path.append("target_model/mld")
from mld.config import parse_args
# from mld.datasets.get_dataset import get_datasets
from mld.data.get_data import get_datasets
from mld.models.get_model import get_model
from mld.utils.logger import create_logger
from generate import generate
import torch


def get_models(device):
    # parse options
    cfg = parse_args(phase="demo")
    cfg.FOLDER = cfg.TEST.FOLDER
    cfg.Name = "demo--" + cfg.NAME
    cfg.DEVICE = [device]
    logger = create_logger(cfg, phase="demo")

    
    # load dataset to extract nfeats dim of model
    dataset = get_datasets(cfg, logger=logger, phase="test")[0]

    # create mld model
    model = get_model(cfg, dataset)


    # loading checkpoints
    # logger.info("Loading checkpoints from {}".format(cfg.TEST.CHECKPOINTS))
    state_dict = torch.load(cfg.TEST.CHECKPOINTS,
                            map_location="cpu")["state_dict"]


    model.load_state_dict(state_dict, strict=True)

    # logger.info("model {} loaded".format(cfg.model.model_type))
    model.sample_mean = cfg.TEST.MEAN
    model.fact = cfg.TEST.FACT
    model.to(device)
    model.eval()

    return model, cfg


def generate_motion(model, cfg, text, output_dir, device, length, render):
    # model = Generate(device)

    # model, cfg = get_models(device)
    motion = generate(
        model, cfg, text, output_dir, device, length, render
    )
    # save_motion(motion, motion_path)

def mdm_gen_motion_from_text(model, cfg, ori_sent, ori_motion_path, device, length, render=False):
    
    generate_motion(model, cfg, ori_sent, ori_motion_path, device, length, render)
    
    return


if __name__ == '__main__':
    ori_sent = "The person walked forward and is picking up his toolbox."
    ori_motion_path = "target_model/mld/target.npy"

    device = 7
    model, cfg = get_models(device)
    mdm_gen_motion_from_text(model, cfg, ori_sent, ori_motion_path, device, length=196)