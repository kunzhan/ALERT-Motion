import numpy as np
import re,os
os.environ["NCCL_P2P_DISABLE"] = "1"
from datetime import datetime
import sys
import random
import shutil
from nltk.tokenize import RegexpTokenizer, word_tokenize
from english import ENGLISH_FILTER_WORDS
from compute_motion_sim import compute_motion_sim
from attack_lib import attack
from attack_lib import sort_words_by_importance, sentence_distance
sys.path.remove('target_model/mdm')
sys.path.append('target_model/mld')
from motion_from_texts import get_models, mdm_gen_motion_from_text
import argparse
from text_new_mld import init_sentence
import multiprocessing
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
# from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model
# from torch.cuda.amp import autocast
from GPT import get_completion
import torch, transformers

total_tokens = 0

model_id = "/data/shared_folder/Meta-Llama-3-8B-Instruct/"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def check_if_contains(tokens):
    flag = False
    loc = 0
    for token in tokens:
        if "_" in token:
            flag = True
            break
        loc += 1
    return flag, loc

def check_if_in_list(sent, sent_ls):
    flag = False
    for tar_sent in sent_ls:
        if sent == tar_sent:
            flag = True
            break
    return flag


def generate(instruction, top_p=0.7, num=1):

    # I want you to work as a writing teacher. You need to study the given sentence, change only a few words from the original sentence according to the requirements and maintain the flow of the sentence. 
    # input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:".format(instruction)
    # input_prompt = "My requirement is {} ".format(instruction)
    input_prompt = instruction
    # if top_p==0.8:
    #     input_prompt = input_prompt + "Please tell me only the revised sentence."
    if num==1:
        output_text, num_tokens = get_completion(input_prompt, top_p, num, model=pipeline)
    else:
        input_prompts = instruction
        output_text, num_tokens = get_completion(input_prompts, top_p, num, model=pipeline)

    # total_tokens = total_tokens + num_tokens
    # while(len(output_text)<10):
    #     output_text = get_completion(input_prompt, top_p)
    return output_text


def get_new_pop(elite_pop, elite_pop_scores, pop_size):
    scores_logits = np.exp(elite_pop_scores - elite_pop_scores.max()) 
    elite_pop_probs = scores_logits / scores_logits.sum()

    s1 = [elite_pop[i] for i in np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    s2 = [elite_pop[i] for i in np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]

    # ----------------------------------------------------------------
    # 种群内部存在长度不一致的情况
    # ----------------------------------------------------------------
    # cand1_len = np.array([len(p1) for p1 in cand1])
    # cand2_len = np.array([len(p2) for p2 in cand1])
    
    instructions = []
    for i in range(pop_size):
        # instruction = "Please take half of each of the following sentences, add any necessary words, and create a new sentence by combining them: {} and {}".format(cand1[i], cand2[i])
        instruction = f"""Given the following two parent prompts which come after the <prompt> tag,
        create a new prompt by crossing over or combining portions of the parents. 
        The new prompt should convey the same as the parents.
        Your new prompt should only contain single quotes and should not include the <prompt> tag.
        
        <prompt>
        Prompt 1: {s1[i]}
        
        Prompt 2: {s2[i]}
        """
        instructions.append(instruction)

    # cand1 = [s1[i].split(' ') for i in range(pop_size)]
    # cand2 = [s2[i].split(' ') for i in range(pop_size)]

    # #exchange two parts randomly
    # position = [[]] * pop_size
    # for i in range(pop_size):
    #     for j in range(len(cand1[i])):
    #         for k in range(len(cand2[i])):
    #             if cand1[i][j] == cand2[i][k] and abs(j-k)<2:
    #                 position[i].append([j, k])
    
    # next_pop = []
    # pop_index = 0
    # instructions = []
    # for pop_flag in position:
    #     # pop = []
    #     # word_index = 0
    #     # for word_flag in pop_flag:
    #     #     if word_flag:
    #     #         try:
    #     #             pop.append(cand1[pop_index][word_index])
    #     #         except ValueError:
    #     #             pop.append(cand2[pop_index][word_index])
    #     #     else:
    #     #         try:
    #     #             pop.append(cand2[pop_index][word_index])
    #     #         except ValueError:
    #     #             pop.append(cand1[pop_index][word_index])
    #     #     word_index += 1
    #     # next_pop.append(pop)
    #     index = np.random.choice(len(pop_flag), size=1)
    #     p = pop_flag[index.item()]
    #     p1_index, p2_index = p[0], p[1]
    #     # pop = cand1[pop_index][:p1_index] + cand2[pop_index][p2_index:]
    #     t1 = ' '.join(cand1[pop_index][:p1_index])
    #     t2 = ' '.join(cand2[pop_index][p2_index:])
    #     instruction = "My requirement is to restructure a complete, flow and coherence sentence by combining two following clauses: '{}' and '{}'.".format(t1, t2)
    #     instructions.append(instruction)
    #     pop_index += 1
        
    next_pop = generate(instructions, num=len(instructions))
    # next_pop.append(pop)

    return next_pop

class Genetic():
    
    def __init__(self, ori_sent, tar_motion_path, tar_sent, log_save_path, intem_motion_path, best_motion_path, mutate_by_impor, model, cfg, length, device):
        
        self.init_pop_size = 15# 150
        self.pop_size = 20# 15
        self.elite_size = 5 # 种群优异个体数 8
        self.mutation_p = 0.85# 0.85
        self.mu = 0.99
        self.alpha = 0.001
        self.max_iters = 50# 50
        self.store_thres = 80

        self.target_motion_path = tar_motion_path
        self.log_save_path = log_save_path
        self.intermediate_path = intem_motion_path
        self.best_motion_path = best_motion_path
        self.target_sent = tar_sent
        self.ori_sent = ori_sent
        self.mutate_by_impor = mutate_by_impor
        
        #initialize attack class
        self.device = device
        # model, cfg = get_models(self.device)
        self.model = model
        self.cfg = cfg
        self.length = length
        self.attack_cls = attack(self.target_sent, self.device)
        # self.tokenizer = tokenizer
        # LlamaTokenizer.from_pretrained("1/InstructZero/WizardLM-13B-V1.2", tokenizer_args={"do_basic_tokenize": False})
        
        # model_path = "1/InstructZero/WizardLM-13B-V1.2"
        # config = LlamaConfig.from_pretrained(model_path)
        # with init_empty_weights():
        #     model = LlamaForCausalLM._from_config(config) 
        # no_split_module_classes = LlamaForCausalLM._no_split_modules
        # device_map = infer_auto_device_map(model, max_memory = { 0: "0.1MIB", 1: "0.1MIB", 2: "20.0GIB", 3: "20.0GIB", 4: "20.0GIB", 5: "20.0GIB", 6: "20.0GIB", 7: "20.0GIB"}, no_split_module_classes=no_split_module_classes) #自动划分每个层的设备
        # # model = LlamaForCausalLM.from_pretrained("1/InstructZero/WizardLM-13B-V1.2", trust_remote_code=False, device_map="balanced")
        # load_checkpoint_in_model(model, model_path, device_map=device_map) #加载权重
        # model = dispatch_model(model,device_map=device_map) #并分配到具体的设备上

        # print(model.get_memory_footprint())
        # 多GPU并行
        # max_memory = get_balanced_memory(
        #     model,
        #     max_memory=None,
        #     no_split_module_classes=["DecoderLayer", "Attention", "MLP", "LayerNorm", "Linear"],
        #     # dtype='float16',
        #     low_zero=False,
        # )
        # max_memory = { 0: "0.0GIB", 1: "10.0GIB", 2: "24.0GIB", 3: "24.0GIB", 4: "24.0GIB", 5: "24.0GIB", 6: "24.0GIB", 7: "24.0GIB"}

        # device_map = infer_auto_device_map(
        #     model,
        #     max_memory=max_memory,
        #     no_split_module_classes=["DecoderLayer", "Attention", "MLP", "LayerNorm", "Linear"],
        #     # dtype='float16'
        # )

        # model = dispatch_model(model, device_map=device_map)
        # self.model = model# .half()# .to(7)
        
        #initialize tokenizer
        # self.tokenizer = RegexpTokenizer(r'\w+')
        # tokens = word_tokenize(ori_sent)# self.tokenizer.tokenize(ori_sent.lower())     

        #generate large initialization corpus
        self.pop = self.initial_mutate(ori_sent, self.init_pop_size)
        print("initial pop: ", self.pop)

    

    def initial_mutate(self, pop, nums):
        #random select the pop sentence that will mutate 
        new_sent_ls = []
        sentence = pop
        
        instruction = self.attack_cls.selectBug(sentence, nums=nums-1, if_initial=True)
        output_texts = generate(instruction)# self.device

        # 使用正则表达式匹配带有序号的句子
        numbered_sentences = re.split(r'\d+\.\s+', output_texts)

        # 如果找到带序号的句子，则去除序号并存储在列表中
        if numbered_sentences:
            new_sent_ls = [sentence.strip() for sentence in numbered_sentences if sentence]
        else:
            # 如果段落中没有带序号的句子，则直接存储句子在列表中
            new_sent_ls = [sentence.strip()+'.' for sentence in output_texts.split('.') if sentence.strip()]

        # sentences = output_texts.split(". ")
        # new_sent_ls = [sentence.strip() for sentence in sentences]
        new_sent_ls.append(pop)
        return new_sent_ls


    def get_fitness_score(self, sentences):
        #get fitness score of all the sentences
        sim_score_ls = []
        
        # model, cfg = get_models(device)

        for i,sentence in enumerate(sentences):
            flag = False
            for j in range(i):
                if sentence==sentences[j]:
                    flag = True
                    break
            if flag:
                sim_score_ls.append(sim_score_ls[j])
                continue
            x_prime_sent = sentence
            # x_prime_sent = x_prime_sent.replace("_", " ")
            
            if sentence_distance(x_prime_sent, self.target_sent)>0.4:
                x_motion_path = self.intermediate_path + "gen.npy"
                # ****************************
                # 将此处的图片生成与相似度度量换成对应的motion
                mdm_gen_motion_from_text(self.model, self.cfg, x_prime_sent, x_motion_path, self.device, self.length)# , self.length

                similarity = compute_motion_sim(x_motion_path, self.target_motion_path, self.device)
                # *****************************
                
                # if similarity > self.store_thres:
                #    best_ori_path = self.best_motion_path + x_prime_sent + "_score_" + str(similarity.item()) + ".png"
                #    shutil.copy(x_motion_path, best_ori_path)

                sim_score_ls.append(similarity.item())

                print(f"x_prime_sent: {x_prime_sent}, similarity: {similarity.item()}")
            else:
                sim_score_ls.append(0)
        sim_score_arr = np.array(sim_score_ls)
        return sim_score_arr
    
    def mutate_pop(self, pop, mutation_p, mutate_by_impor):
        #random select the pop sentence that will mutate
        mask = np.random.rand(len(pop)) < mutation_p 
        new_pop = []
        pop_index = 0
        # if mutate_by_impor:
        #     x_prime_sent = " ".join(pop[pop_index])
        #     sim_probs = sort_words_by_importance(self.intermediate_path, x_prime_sent, self.target_motion_path)
        # else:
        #     sim_probs = np.full(min(len(tokens), 20), 1/min(len(tokens), 20))
        sentences = []
        for flag in mask:
            if not flag:
                new_pop.append(pop[pop_index])
            else:
                sentence = pop[pop_index]
                sentences.append(sentence)

        instructions = [self.attack_cls.selectBug(sentence) for sentence in sentences]
        output_text = generate(instructions, num=len(instructions))


        new_pop = output_text
            # pop_index += 1
        
        return new_pop
                    
    def run(self, log=None):
        best_save_dir = self.best_motion_path
        itr = 1
        prev_score = None
        save_dir = self.intermediate_path
        best_score = float("-inf")
        if log is not None:
            log.write(f'init_pop_size: {self.init_pop_size}\t pop_size: {self.pop_size}\t elite_size: {self.elite_size} \n')
            log.write(f'mutation_p: {self.mutation_p}\t max_iters: {self.max_iters}\n')
            log.write('target phrase: ' + self.target_sent + '\n')
            log.write('ori phrase:'+ self.ori_sent + '\n')
            log.flush()
        
        while itr <= self.max_iters:
            print("\n")
            print(f"----------------------------------------")
            print(f"-----------itr num:{itr}----------------")
            print(f"----------------------------------------")
            print("\n")
            log.write("------------- iteration:" + str(itr) + " ---------------\n")
            pop_scores = self.get_fitness_score(self.pop)
            elite_ind = np.argsort(pop_scores)[-self.elite_size:]
            elite_pop = [self.pop[i] for i in elite_ind]
            elite_pop_scores = pop_scores[elite_ind]

            print("current best score: ", elite_pop_scores[-1])
            
            for i in elite_ind:
                x_prime_sent_store = self.pop[i]
                # x_prime_sent_store = x_prime_sent_store.replace("_", " ")
                if pop_scores[i] > self.store_thres and sentence_distance(x_prime_sent_store, self.target_sent)>0.4:
                    log.write(str(pop_scores[i]) + " " + x_prime_sent_store + "\n")
            
            if elite_pop_scores[-1] > best_score:
                best_score = elite_pop_scores[-1]
                #store the current best image
                x_prime_sent = elite_pop[-1]
                # x_prime_sent = x_prime_sent.replace("_", " ")
                
                x_motion_path_npy = save_dir + "gen.npy"
                x_motion_path_mp4 = save_dir + "gen.mp4"

                mdm_gen_motion_from_text(self.model, self.cfg, x_prime_sent, x_motion_path_npy, self.device, self.length)# , self.length

                best_ori_path_npy = best_save_dir + "/itr_" + str(itr) + "_score_" + str(elite_pop_scores[-1]) + ".npy"
                best_ori_path_mp4 = best_save_dir + "/itr_" + str(itr) + "_score_" + str(elite_pop_scores[-1]) + ".mp4"
                
                shutil.copy(x_motion_path_npy, best_ori_path_npy)
                shutil.copy(x_motion_path_mp4, best_ori_path_mp4)

                final_best, tail = os.path.split(best_save_dir)
                best_ori_path_npy = final_best + "/" + tail + ".npy"
                best_ori_path_mp4 = final_best + "/" + tail + ".mp4"
                
                shutil.copy(x_motion_path_npy, best_ori_path_npy)
                shutil.copy(x_motion_path_mp4, best_ori_path_mp4)

                #new best adversarial sentences
                log.write("new best adv: " +  str(elite_pop_scores[-1]) + " " + x_prime_sent + "\n")
                

            
            if prev_score is not None and prev_score != elite_pop_scores[-1]: 
                self.mutation_p = self.mu * self.mutation_p + self.alpha / np.abs(elite_pop_scores[-1] - prev_score) 

            next_pop = get_new_pop(elite_pop, elite_pop_scores, self.pop_size)

            self.pop = self.mutate_pop(next_pop, self.mutation_p, self.mutate_by_impor)
            
            prev_score = elite_pop_scores[-1]
            itr += 1
            log.flush()

        return 


def run_attack(device):

    current_time = datetime.now()
    v_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    for folder in ['intermediate_motion_path', 'best_motion_path']:
        folder_name = f"attack/{folder}_{v_str}"
        os.makedirs(folder_name)

    # device = device

    ori_sent="The camera pans across an empty field, showing trees in various states of growth."
    tar_sent="The person walked forward and is picking up his toolbox."
    tar_motion_path="attack/target.npy"
    top_k=4
    text_length=15
    log_save_path=f'attack/run_log_{v_str}.txt'
    intem_motion_path=f'attack/intermediate_motion_path_{v_str}/'
    best_motion_path=f'attack/best_motion_path_{v_str}/'
    mutate_by_impor=False


    model, cfg = get_models(device)
    length = (np.load(tar_motion_path)).shape[3]
    with open(log_save_path, 'w') as log:
        log.write(f'text_length: {text_length}\t top_k: {top_k}\n')

    return v_str, "a"#init_sentence(text_length, top_k, intem_motion_path, best_motion_path, tar_motion_path, device, model, cfg, length)



def run_attacks(i, id_list, v_str):
    # # 获取当前时间
    # current_time = datetime.now()

    # # 格式化时间戳
    # v_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    # v_str = "2023-10-27_16-32-00"
    id_str = id_list[i]

    # 创建文件夹
    for folder in ['intermediate_motion_path', 'best_motion_path']:
        folder_name = f"attack/{folder}_{v_str}/{id_str}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    directory_path = f"attack/run_log_{v_str}"
    if not os.path.exists(directory_path):
    # 如果目录不存在，使用os.makedirs()创建它
        os.makedirs(directory_path)
    # os.makedirs(f"attack/run_log_{v_str}")


    # model_path = "1/InstructZero/WizardLM-13B-V1.2"
    # tokenizer = LlamaTokenizer.from_pretrained(model_path, tokenizer_args={"do_basic_tokenize": False})
    # config = LlamaConfig.from_pretrained(model_path)
    # with init_empty_weights():
    #     model = LlamaForCausalLM._from_config(config) 
    # no_split_module_classes = LlamaForCausalLM._no_split_modules
    # device_map = infer_auto_device_map(model, max_memory = { 0: "0.1MIB", 1: "10.0GIB", 2: "10.0GIB", 3: "10.0GIB", 4: "10.0GIB", 5: "10.0GIB", 6: "10.0GIB", 7: "10.0GIB"}, no_split_module_classes=no_split_module_classes) #自动划分每个层的设备
    # load_checkpoint_in_model(model, model_path, device_map=device_map) #加载权重
    # model = dispatch_model(model,device_map=device_map) #并分配到具体的设备上


    device = i%4

    text = id_str+'.txt'
    text_file = f"target_model/mdm/datasets/HumanML3D/texts/{text}"
    with open(text_file, 'r') as file:
        tar_sent = ''
        # 逐行读取文件
        for line in file:
            for char in line:
                if char == '#':
                    break  # 如果遇到'#'字符，退出内层循环
                tar_sent += char  # 将字符添加到变量
            break

    gt_motion_path = f"target_model/mld/datasets/HumanML3D/new_joint_vecs/{id_str}.npy"
    model, cfg = get_models(device)
    length = (np.load(gt_motion_path)).shape[0]
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--ori_sent', type=str, required=False, help='original sentence', default="The camera pans across an empty field, showing nothing happening in particular as a person stumbles around.")
    # parser.add_argument('--tar_sent', type=str, required=False, help='target sentence', default=tar_sent)
    tar_motion_path = f"attack/target/{id_str}.npy"
    mdm_gen_motion_from_text(model, cfg, tar_sent, tar_motion_path, device, length)


    # parser = argparse.ArgumentParser()
    # # parser.add_argument('--ori_sent', type=str, required=False, help='original sentence', default="A woman walks slowly across the beach towards her car while her dog follows closely behind.")
    # parser.add_argument('--tar_sent', type=str, required=False, help='target sentence', default="The person walked forward and is picking up his toolbox.")

    # parser.add_argument('--tar_motion_path', type=str, required=False, help='target motion path', default="attack/target.npy")
    # parser.add_argument('--top_k', type=int, required=False, help='top k', default=4)
    # parser.add_argument('--text_length', type=int, required=False, help='text_length', default=15)
    
    # parser.add_argument('--log_save_path', type=str, default=f'attack/run_log_{v_str}/{id_str}.txt', help='path to save log')
    # parser.add_argument('--intem_motion_path', type=str, default=f'attack/intermediate_motion_path_{v_str}/{id_str}/', help='path to save intermediate motions')
    # parser.add_argument('--best_motion_path', type=str, default=f'attack/best_motion_path_{v_str}/{id_str}/', help='path to save best output motions')
    # parser.add_argument('--mutate_by_impor', type=bool, default=False, help='whether select word by importance in mutation')
    # args = parser.parse_args()
    top_k = 8
    text_length = 20
    log_save_path = f'attack/run_log_{v_str}/{id_str}.txt'
    intem_motion_path = f'attack/intermediate_motion_path_{v_str}/{id_str}/'
    best_motion_path = f'attack/best_motion_path_{v_str}/{id_str}'
    mutate_by_impor = False

    with open(log_save_path, 'w') as log:
        log.write(f'text_length: {text_length}\t top_k: {top_k}\n')

    import json
    with open('attack/ori_sentences_mld.json', 'r') as f:
        ori_sentences = json.load(f)
    ori_sent = ori_sentences[i]
    g = Genetic(ori_sent, tar_motion_path, tar_sent, log_save_path, intem_motion_path, best_motion_path, mutate_by_impor, model, cfg, length, device)
    # g = Genetic(init_sentence(text_length, top_k, intem_motion_path, best_motion_path, tar_motion_path, device, model, cfg, length), tar_motion_path, tar_sent, log_save_path, intem_motion_path, best_motion_path, mutate_by_impor, model, cfg, length, device)
    # g = Genetic(args.ori_sent, args.tar_motion_path, args.tar_sent, args.log_save_path, args.intem_motion_path, args.best_motion_path, args.mutate_by_impor, model, cfg, length)
    # g = Genetic(get_completion("My requirement is to write a sentence of no more than 30 words about human movement scenes description.", top_p=0.8), tar_motion_path, tar_sent, log_save_path, intem_motion_path, best_motion_path, mutate_by_impor, model, cfg, length, device)
    
    with open(log_save_path, 'w') as log:
        g.run(log=log)
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--i', type=int, required=True, default=0)
    # args = parser.parse_args()
    # start = args.i*2
    # end = args.i*2+2
    # for i in range(start, end):
    #     # 格式化时间戳
    #     # current_time = datetime.now()
    #     # timestamp = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    #     device = i//2
    #     v_str = str(i).zfill(3)
    #     # 创建文件夹
    #     for folder in ['intermediate_motion_path', 'best_motion_path']:
    #         folder_name = f"attack/{folder}_{v_str}"
    #         os.makedirs(folder_name)
        
    #     # tar_sents = []
    #     text = str(i).zfill(6)+'.txt'
    #     text_file = f"target_model/mdm/datasets/HumanML3D/texts/{text}"
    #     with open(text_file, 'r') as file:
    #         tar_sent = ''
    #         # 逐行读取文件
    #         for line in file:
    #             for char in line:
    #                 if char == '#':
    #                     break  # 如果遇到'#'字符，退出内层循环
    #                 tar_sent += char  # 将字符添加到变量
    #             break

    #     tar_motion_path_npy = f"attack/target{v_str}.npy"

    #     parser = argparse.ArgumentParser()
    #     # parser.add_argument('--ori_sent', type=str, required=False, help='original sentence', default="The camera pans across an empty field, showing nothing happening in particular as a person stumbles around.")
    #     parser.add_argument('--tar_sent', type=str, required=False, help='target sentence', default=tar_sent)
        
    #     mdm_gen_motion_from_text(tar_sent, tar_motion_path_npy, device)

    #     parser.add_argument('--tar_motion_path', type=str, required=False, help='target motion path', default=tar_motion_path_npy)
    #     parser.add_argument('--top_k', type=int, required=False, help='top k', default=8)
    #     parser.add_argument('--text_length', type=int, required=False, help='text_length', default=15)
        
    #     parser.add_argument('--log_save_path', type=str, default=f'attack/run_log_{v_str}.txt', help='path to save log')
    #     parser.add_argument('--intem_motion_path', type=str, default=f'attack/intermediate_motion_path_{v_str}/', help='path to save intermediate motions')
    #     parser.add_argument('--best_motion_path', type=str, default=f'attack/best_motion_path_{v_str}/', help='path to save best output motions')
    #     parser.add_argument('--mutate_by_impor', type=bool, default=False, help='whether select word by importance in mutation')
    #     args = parser.parse_args()
    #     with open(args.log_save_path, 'w') as log:
    #         log.write(f'text_length: {args.text_length}\t top_k: {args.top_k}\n')

    #     g = Genetic(init_sentence(args.text_length, args.top_k, args.intem_motion_path, args.best_motion_path, args.tar_motion_path, device), args.tar_motion_path, args.tar_sent, args.log_save_path, args.intem_motion_path, args.best_motion_path, args.mutate_by_impor, device)
    #     # g = Genetic(args.ori_sent, args.tar_motion_path, args.tar_sent, args.log_save_path, args.intem_motion_path, args.best_motion_path, args.mutate_by_impor, device)
        
    #     with open(args.log_save_path, 'w') as log:
    #         g.run(log=log)


if __name__ == "__main__":
    run_attacks()