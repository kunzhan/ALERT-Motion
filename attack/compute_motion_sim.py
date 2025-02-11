import torch
import numpy as np
import sys
# import clip
# from PIL import Image
# sys.path.remove('target_model/mdm')
# sys.path.append('target_model/mld')
sys.path.append('target_model/mdm')
sys.path.append('target_model/TMR')
# from utils.model_util import MDM_Model, load_model_wo_clip
# from sample.generate import Generate
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from encode_motions import encode_motion
from encode_texts import encode_text
from models import load_model
from utils import dist_util
from torch import nn

# def compute_motion_sim(motion1_path, motion2_path, device):

#     # device = "cuda:6" if torch.cuda.is_available() else "cpu"

#     motion1 = np.load(motion1_path)
#     motion2 = np.load(motion2_path)
#     motion1 = torch.from_numpy(motion1).squeeze(2).transpose(1,2)
#     motion2 = torch.from_numpy(motion2).squeeze(2).transpose(1,2)
#     motion1 = motion1.to(device)
#     motion2 = motion2.to(device)

#     eval_wrapper = EvaluatorMDMWrapper('humanml', dist_util.dev())
#     with torch.no_grad():
#         motion1_features = eval_wrapper.get_motion_embeddings(motion1, torch.tensor([motion1.shape[1]]))
#         motion2_features = eval_wrapper.get_motion_embeddings(motion2, torch.tensor([motion2.shape[1]]))

#         motion1_features /= motion1_features.norm(dim=-1, keepdim=True)
#         motion2_features /= motion2_features.norm(dim=-1, keepdim=True)
#         similarity = 100. * (motion1_features @ motion2_features.T)
    
#     # loss_fn = nn.MSELoss()
#     # MSE_loss = loss_fn(motion1, motion2)
#     alpha = 1

#     # print("MSE_loss: {}\n".format(MSE_loss.item()))
#     return similarity#-alpha*MSE_loss.item()
    # return torch.tensor([[1]])

# t2m, m2t, m2m - t2t
def compute_motion_sim(motion1_path, motion2_path, device):

    # device = "cuda:6" if torch.cuda.is_available() else "cpu"
    # text = "The person walked forward and is picking up his toolbox."
    motion1 = np.load(motion1_path)
    motion2 = np.load(motion2_path)
    motion1 = torch.from_numpy(motion1).squeeze(2).transpose(1,2)
    motion2 = torch.from_numpy(motion2).squeeze(2).transpose(1,2)
    # motion2 = torch.from_numpy(motion2).unsqueeze(0)

    motion1 = motion1.to(device)
    motion2 = motion2.to(device)

    # eval_wrapper = EvaluatorMDMWrapper('humanml', dist_util.dev())
    model, text_model, normalizer = load_model(device)
    with torch.no_grad():
        # text_feature = encode_text(text, device)
        motion1_feature = encode_motion(motion1, device, model, normalizer)
        motion2_feature = encode_motion(motion2, device, model, normalizer)

        # text_features = text_feature/text_feature.norm(dim=-1, keepdim=True)
        motion1_features = motion1_feature/motion1_feature.norm(dim=-1, keepdim=True)
        motion2_features = motion2_feature/motion2_feature.norm(dim=-1, keepdim=True)
        m2m = 100. * (motion1_features @ motion2_features.T)/2 + 50
        # t2m = 100. * (text_features @ motion2_features.T)/2 + 50
    
    # loss_fn = nn.MSELoss()
    # MSE_loss = loss_fn(motion1, motion2)
    # alpha1 = 0.2
    # alpha2 = 0.2
    # similarity = m2m * (1-alpha1) + t2m * alpha1

    # print("MSE_loss: {}\n".format(MSE_loss.item()))
    return m2m# similarity-alpha2*MSE_loss.item()


if __name__ == '__main__':
    compute_motion_sim("attack_copy/target1.npy", "attack_copy/target.npy", 5)