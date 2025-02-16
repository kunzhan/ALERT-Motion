# [Autonomous LLM-enhanced adversarial attack for text-to-motion](https://arxiv.org/abs/2408.00352)

Human motion generative models have enabled promising applications, but the ability of text-to-motion (T2M) models to produce realistic motions raises security concerns if exploited maliciously. Despite growing interest in T2M, limited research focus on safeguarding these models against adversarial attacks, with existing work on text-to-image models proving insufficient for the unique motion domain. In the paper, we propose ALERT-Motion, an autonomous framework that leverages large language models (LLMs) to generate targeted adversarial attacks against black-box T2M models. Unlike prior methods that modify prompts through predefined rules, ALERT-Motion uses the knowledge of LLMs of human motion to autonomously generate subtle yet powerful adversarial text descriptions. It comprises two key modules: an adaptive dispatching module that constructs an LLM-based agent to iteratively refine and search for adversarial prompts; and a multimodal information contrastive module that extracts semantically relevant motion information to guide the agent's search. Through this LLM-driven approach, ALERT-Motion produces adversarial prompts querying victim models to produce outputs closely matching targeted motions, while avoiding obvious perturbations. Evaluations across popular T2M models demonstrate ALERT-Motion's superiority over previous methods, achieving higher attack success rates with stealthier adversarial prompts. This pioneering work on T2M adversarial attacks highlights the urgency of developing defensive measures as motion generation technology advances, urging further research into safe and responsible deployment.

# Experiment
This repo contains the code of a PyTorch implementation of [Autonomous LLM-enhanced adversarial attack for text-to-motion](https://arxiv.org/abs/2408.00352)
- Target Models
	- [mdm](https://github.com/GuyTevet/motion-diffusion-model)
	- [mld](https://github.com/ChenFengYe/motion-latent-diffusion)
	- [TMR](https://github.com/Mathux/TMR)

Among them, the TMR model [Mathux/TMR](https://github.com/Mathux/TMR) is an embedding model for computing similarity, responsible for calculating the similarity between motions during the attack process. MDM and MLD are the target models to be attacked, and their file structures are as follows:
📦 mdm
 ┣ 📂 assets
 ┣ 📂 body_models
 ┣ 📂 data_loaders
 ┣ 📂 datasets
 ┣ 📂 diffusion
 ┣ 📂 eval
 ┣ 📂 glove
 ┣ 📂 kit
 ┣ 📂 model
 ┣ 📂 prepare
 ┣ 📂 sample
 ┣ 📂 save
 ┣ 📂 t2m
 ┣ 📂 train
 ┣ 📂 utils
 ┗ 📂 visualize

📦 mld
 ┣ 📂 checkpoints
 ┣ 📂 configs
 ┣ 📂 datasets
 ┣ 📂 demo
 ┣ 📂 deps
 ┣ 📂 mld
 ┣ 📂 prepare
 ┣ 📂 results
 ┗ 📂 scripts

📦 TMR
 ┣ 📂 configs
 ┣ 📂 datasets
 ┣ 📂 demo
 ┣ 📂 logs
 ┣ 📂 models
 ┣ 📂 outputs
 ┣ 📂 prepare
 ┣ 📂 src
 ┗ 📂 stats

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd attack
bash run.sh # Run attack on mld model, all_count=20 is the number of examples to attack, usually 20, taking the first 20 examples from target_model\TMR\nsim_test.txt
```



# Citation

We appreciate it if you cite the following paper:

```sh
@InProceedings{miaoAAAI2025,
  author =    {Honglei Miao and Fan Ma and Ruijie Quan and Kun Zhan and Yi Yang},
  title =     {Autonomous {LLM}-enhanced adversarial attack for text-to-motion},
  booktitle = {AAAI},
  year =      {2025},
  volume =    {39},
  number =    {},
  pages =     {--},
}
```

## Contact
https://kunzhan.github.io/

If you have any questions, feel free to contact me. (Email: `ice.echo#gmail.com`)