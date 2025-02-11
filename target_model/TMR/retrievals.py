import os,sys,torch
# sys.path.append("target_model/TMR")
from omegaconf import DictConfig
import logging
import hydra
import yaml
from tqdm import tqdm
sys.path.append('target_model/mdm')
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
from utils import dist_util
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from data_loaders.humanml.utils.metrics import *
logger = logging.getLogger(__name__)


def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)



def compute_sim_matrix(model, dataset, keyids, gt_dataset, batch_size=256):
    import torch
    import numpy as np
    from src.data.collate import collate_text_motion
    from src.model.tmr import get_sim_matrix

    device = model.device

    nsplit = int(np.ceil(len(dataset) / batch_size))
    
    with torch.inference_mode():
        all_data = [dataset.load_keyid(keyid) for keyid in keyids]
        all_data_splitted = np.array_split(all_data, nsplit)

        gt_all_data = [gt_dataset.load_keyid(keyid) for keyid in keyids]
        gt_all_data_splitted = np.array_split(gt_all_data, nsplit)

        # by batch (can be too costly on cuda device otherwise)
        latent_texts = []
        latent_motions = []
        sent_embs = []

        # gt_latent_texts = []
        gt_latent_motions = []
        for data in tqdm(all_data_splitted, leave=False):
            batch = collate_text_motion(data, device=device)

            # eval_wrapper = EvaluatorMDMWrapper('humanml', dist_util.dev())
            # evaluate_matching_score(eval_wrapper, batch)
            # Text is already encoded
            text_x_dict = batch["text_x_dict"]
            motion_x_dict = batch["motion_x_dict"]
            sent_emb = batch["sent_emb"]

            # Encode both motion and text
            latent_text = model.encode(text_x_dict, sample_mean=True)
            latent_motion = model.encode(motion_x_dict, sample_mean=True)
            
            # eval_wrapper = EvaluatorMDMWrapper('humanml', dist_util.dev())
            # latent_text, latent_motion = eval_wrapper.get_co_embeddings(
            #     word_embs=text_x_dict["word_embeddings"],
            #     pos_ohot=text_x_dict["pos_one_hots"],
            #     cap_lens=text_x_dict["length"],
            #     motions=motion_x_dict["x"],
            #     m_lens=motion_x_dict["length"]
            # )

            latent_texts.append(latent_text)
            latent_motions.append(latent_motion)
            sent_embs.append(sent_emb)

        latent_texts = torch.cat(latent_texts)
        latent_motions = torch.cat(latent_motions)

        for data in tqdm(gt_all_data_splitted, leave=False):
            batch = collate_text_motion(data, device=device)

            # Text is already encoded
            # text_x_dict = batch["text_x_dict"]
            motion_x_dict = batch["motion_x_dict"]
            # sent_emb = batch["sent_emb"]

            # Encode both motion and text
            # latent_text = model.encode(text_x_dict, sample_mean=True)
            latent_motion = model.encode(motion_x_dict, sample_mean=True)

            # gt_latent_texts.append(latent_text)
            gt_latent_motions.append(latent_motion)

        # gt_latent_texts = torch.cat(gt_latent_texts)
        gt_latent_motions = torch.cat(gt_latent_motions)

        # 计算语义距离
        dist_mat = euclidean_distance_matrix(latent_texts.cpu().numpy(),
                                                latent_motions.cpu().numpy())
        matching_score_sum = dist_mat.trace()
        matching_score = matching_score_sum / latent_texts.shape[0]

        # 计算FID
        gt_mu, gt_cov = calculate_activation_statistics(gt_latent_motions.cpu().numpy())
        mu, cov = calculate_activation_statistics(latent_motions.cpu().numpy())
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

        sent_embs = torch.cat(sent_embs)
        sim_matrix = get_sim_matrix(latent_texts, latent_motions)
    returned = {
        "sim_matrix": sim_matrix.cpu().numpy(),
        "MM_dist": matching_score,
        "FID": fid,
        "sent_emb": sent_embs.cpu().numpy(),
    }
    return returned


def main_retrieval(data_path, split, device=7):
    @hydra.main(version_base=None, config_path="configs", config_name="retrieval")
    def retrieval(newcfg):
        protocol = "nsim"
        # threshold_val = newcfg.threshold
        # device = newcfg.device
        run_dir = "target_model/TMR/models/tmr_humanml3d_guoh3dfeats/"
        ckpt_name = "last"
        batch_size = 256

        assert protocol in ["all", "normal", "threshold", "nsim", "guo"]

        if protocol == "all":
            protocols = ["normal", "threshold", "nsim", "guo"]
        else:
            protocols = [protocol]

        save_dir = os.path.join(run_dir, "contrastive_metrics")
        os.makedirs(save_dir, exist_ok=True)

        # Load last config
        from src.config import read_config
        import src.prepare  # noqa

        cfg = read_config(run_dir)

        import pytorch_lightning as pl
        import numpy as np
        from hydra.utils import instantiate
        from src.load import load_model_from_cfg
        from src.model.metrics import all_contrastive_metrics, print_latex_metrics

        pl.seed_everything(cfg.seed)

        logger.info("Loading the model")
        model = load_model_from_cfg(cfg, ckpt_name, eval_mode=True, device=device)


        datasets = {}
        results = {}
        for protocol in protocols:
            # Load the dataset if not already
            if protocol not in datasets:
                if protocol in ["normal", "threshold", "guo"]:
                    dataset = instantiate(cfg.data, split="test")
                    datasets.update(
                        {key: dataset for key in ["normal", "threshold", "guo"]}
                    )
                elif protocol == "nsim":
                    datasets[protocol] = instantiate(cfg.data, data_path=data_path, split=split)
            dataset = datasets[protocol]


            # Compute sim_matrix for each protocol
            if protocol not in results:
                if protocol in ["normal", "threshold"]:
                    res = compute_sim_matrix(
                        model, dataset, dataset.keyids, batch_size=batch_size
                    )
                    results.update({key: res for key in ["normal", "threshold"]})
                elif protocol == "nsim":
                    gt_dataset = instantiate(cfg.data, split=split)
                    res = compute_sim_matrix(
                        model, dataset, dataset.keyids, gt_dataset, batch_size=batch_size
                    )
                    results[protocol] = res
                elif protocol == "guo":
                    keyids = sorted(dataset.keyids)
                    N = len(keyids)

                    # make batches of 32
                    idx = np.arange(N)
                    np.random.seed(0)
                    np.random.shuffle(idx)
                    idx_batches = [
                        idx[32 * i : 32 * (i + 1)] for i in range(len(keyids) // 32)
                    ]

                    # split into batches of 32
                    # batched_keyids = [ [32], [32], [...]]
                    results["guo"] = [
                        compute_sim_matrix(
                            model,
                            dataset,
                            np.array(keyids)[idx_batch],
                            batch_size=batch_size,
                        )
                        for idx_batch in idx_batches
                    ]
            result = results[protocol]

            # Compute the metrics
            if protocol == "guo":
                all_metrics = []
                for x in result:
                    sim_matrix = x["sim_matrix"]
                    metrics = all_contrastive_metrics(sim_matrix, rounding=None)
                    all_metrics.append(metrics)

                avg_metrics = {}
                for key in all_metrics[0].keys():
                    avg_metrics[key] = round(
                        float(np.mean([metrics[key] for metrics in all_metrics])), 2
                    )

                metrics = avg_metrics
                protocol_name = protocol
            else:
                sim_matrix = result["sim_matrix"]

                protocol_name = protocol
                if protocol == "threshold":
                    emb = result["sent_emb"]
                    threshold = threshold_val
                    protocol_name = protocol + f"_{threshold}"
                else:
                    emb, threshold = None, None
                metrics = all_contrastive_metrics(sim_matrix, emb, threshold=threshold)

            print_latex_metrics(metrics)

            metric_name = f"{protocol_name}.yaml"
            path = os.path.join(save_dir, metric_name)
            save_metric(path, metrics)

            logger.info(f"Testing done, metrics saved in:\n{path}")
    retrieval()

if __name__ == "__main__":
    data_path = "attack/mdm_best_motion_path_2024-01-12_11-47-37/"
    split = "nsim_test1"
    main_retrieval(data_path, split)
