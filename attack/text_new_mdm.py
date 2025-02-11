from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList, \
    STOPPING_CRITERIA_INPUTS_DOCSTRING, add_start_docstrings
import torch,sys
import os,shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from compute_motion_sim import compute_motion_sim
# sys.path.remove('target_model/mdm')
sys.path.append('target_model/mdm')
from motion_from_text import mdm_gen_motion_from_text
import numpy as np



class StopAtSpecificTokenCriteria(StoppingCriteria):
    """
    当生成出第一个指定token时, 立即停止生成
    """
    def __init__(self, token_id_list):
        """
        :param token_id_list: 停止生成的指定token的id的列表
        """
        self.token_id_list = token_id_list
        
    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # return np.argmax(scores[-1].detach().cpu().numpy()) in self.token_id_list
        # 储存scores会额外占用资源，所以直接用input_ids进行判断
        return input_ids[0][-1].detach().cpu().numpy() in self.token_id_list



class MotionScoresTokenLogitsProcessor(LogitsProcessor):
    """
    综合考虑token对文本流畅和其生成运动与目标运动相似度的影响
    """
    def __init__(self,  input_len: int, top_k: int,
                tar_motion_path: str, tar_sentence: str, intem_motion_path: str,
                best_motion_path: str, tokenizer, device, model, cfg, bad_words_set, length):
        """
        :param  ori_sentence: 原始的句子
                input_len: 原始句子的长度
                top_k: 每轮选取前k个GPT打分最高的单词, 计算其生成运动的相似度
                tar_motion_path: 目标运动文件路径
                tar_sentence: 目标运动对应的句子
                intem_motion_path: 暂存生成运动文件路径
                best_motion_path: 存放每次K个单词中相似度最高的运动
        """
        # self.ori_sentence = ori_sentence
        self.input_len = input_len
        self.top_k = top_k
        self.target_motion_path = tar_motion_path
        self.target_sent = tar_sentence
        # self.log_save_path = log_save_path
        self.intermediate_path = intem_motion_path
        self.best_motion_path = best_motion_path
        self.tokenizer = tokenizer
        self.device = device
        self.model = model
        self.cfg = cfg
        self.bad_words_set = bad_words_set
        self.length = length
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        itr = input_ids.shape[1] - self.input_len
        # log.write("------------- origin sentence init: ---------------\n")
        
        if input_ids.shape[1] == self.input_len:
            generated_words = torch.tensor([])
        else:
            generated_words = input_ids[0,self.input_len:]
        sorted_scores, ids = torch.sort(scores, descending=True)
        # next_words = tokenizer.convert_ids_to_tokens(ids[0,:top_k])
        next_words = []
        for i in range(self.top_k):
            next_words.append(ids[0,i])
        x_motion_path = self.intermediate_path + "gen.npy"
        sim_score_ls = []
        new_scores = torch.full(scores.shape, -float("Inf"))
        for i,next_word in enumerate(next_words):
            # if list(next_word)[0] == 'Ġ':
            #     sentence = generated_words + ' ' + ''.join(list(next_word)[1:])
            # else:
            #     sentence = generated_words + next_word
            # if next_word=='.' and itr < 15:
            #     sim_score_ls.append(torch.tensor(-float("Inf")))
            # else:
            if next_word.item() in self.bad_words_set:
                sim_score_ls.append(0)
                continue

            final_words = np.append(generated_words.numpy(), next_word.reshape(1).numpy())
            sentence = self.tokenizer.decode(torch.from_numpy(final_words.astype(np.int64)))
            mdm_gen_motion_from_text(self.model, self.cfg, sentence, x_motion_path, self.device)# , self.length
            similarity = compute_motion_sim(x_motion_path, self.target_motion_path, self.device)
            sim_score_ls.append(similarity.item())
            print(f"x_prime_sent: {sentence}, similarity: {similarity.item()}")
            # log.write(str(similarity.item()) + " " + sentence + "\n")

        sim_score = torch.FloatTensor(sim_score_ls)
        sim_z_score = (sim_score-sim_score.mean())/sim_score.std()
        for j in range(len(next_words)):
            # scores[:,ids[0,j]] = scores[:,ids[0,j]]+sim_z_score[j]
            if ids[0,j]==13:
                if  itr < 15:
                    continue
                else:
                    new_scores[:,ids[0,j]] = sim_z_score[j] + (2)/sim_score.std()
            else:
                new_scores[:,ids[0,j]] = sim_z_score[j] + 0.001*scores[:,ids[0,j]]
        
        # global best_score
        # if sim_score.max() > best_score:
        # best_score = sim_score.max()
        #store the current best motion
        # sentence = generated_words + next_words[sim_score.argmax()]
        
        # x_motion_path_npy = self.intermediate_path + "gen.npy"
        # x_motion_path_mp4 = self.intermediate_path + "gen.mp4"

        # mdm_gen_motion_from_text(sentence, x_motion_path_npy)

        # best_ori_path_npy = self.best_motion_path + "itr_" + str(itr) + "_score_" + str(best_score.item()) + ".npy"
        # best_ori_path_mp4 = self.best_motion_path + "itr_" + str(itr) + "_score_" + str(best_score.item()) + ".mp4"
        
        # shutil.copy(x_motion_path_npy, best_ori_path_npy)
        # shutil.copy(x_motion_path_mp4, best_ori_path_mp4)

        # log.write("new best adv: " +  str(sim_score.max()) + " " + sentence + "\n")
        # log.flush()
        # global i
        # i = i+1
        return new_scores
# sim_score = torch.zeros(5)
# value1, value2 = sim_score.max(), sim_score.argmax()
# next_word = 'Ġperson'
# ''.join(c=' ' if c=='Ġ' for c in list(next_word))

# global best_score 
# best_score = float("-inf")
# global i
# i = 0

def init_sentence(text_length, top_k, intem_motion_path, best_motion_path, tar_motion_path, device, model, cfg, length):
    top_k = top_k
    # tar_motion_path = "/attack/target.npy"
    # intem_motion_path = '/attack/intermediate_motion_path_10/'
    # best_motion_path = '/attack/best_motion_path_10/'
    # log_save_path = 'run_log_10.txt'

    # device  = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("/LaMini-GPT-1.5B", tokenizer_args={"do_basic_tokenize": False})
    text_model = AutoModelForCausalLM.from_pretrained("/LaMini-GPT-1.5B", trust_remote_code=False)

    # ori_sentence = 'A man kicks something or someone with his right leg.'
    tar_sentence = 'The person walked forward and is picking up his toolbox.'

    # instruction = 'Continue writing a paragraph about using the third person and describing human motion scenes that begins with the following sentence: {}'.format(ori_sentence)
    instruction = ''
    input_prompt = "Please write a sentence about human motion scene description:"

    # ids = tokenizer.convert_tokens_to_ids(['.'])
    # tokens = tokenizer.convert_ids_to_tokens([13])
    input_ids = tokenizer.encode(input_prompt, return_tensors="pt")

    _, input_len = input_ids.shape

    # tmp = model.forward(input_ids)
    # a, idx1 = torch.sort(tmp.logits[:,-1,:], descending=True)#descending为alse，升序，为True，降序
    # idx = idx1[0,:8]
    # next_words = tokenizer.convert_ids_to_tokens(idx)
    logits_processor = LogitsProcessorList()
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(StopAtSpecificTokenCriteria(token_id_list=[13]))


    # with open(log_save_path, 'w') as log:
        # if log is not None:
        #     log.write('target phrase: ' + tar_sentence + '\n')
        #     log.write('origin phrase:'+ ori_sentence + '\n')
        # log.write("------------- origin sentence init: ---------------\n")
    bad_words = ['\"', "\'", '\n']
    bad_words_ids = tokenizer(bad_words).input_ids 
    bad_words_set = set([bad_words_ids[i][0] for i in range(len(bad_words_ids))])
    bad_words_set.add(366)

    logits_processor.append(MotionScoresTokenLogitsProcessor(input_len, top_k, tar_motion_path, tar_sentence, intem_motion_path, best_motion_path, tokenizer, device, model, cfg, bad_words_set, length))
    # output = model.generate(input_ids, num_beams=3, do_sample=False, min_length=text_length+input_len, max_length=text_length+input_len, logits_processor=logits_processor)# , temperature=0.7

    output = text_model.generate(input_ids, min_length=15+input_len, max_length=35+input_len, logits_processor=logits_processor, stopping_criteria=stopping_criteria, bad_words_ids=bad_words_ids)# , temperature=0.7
        # log.flush()
        
    output_text = tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
    print('init sentence: ', output_text)
    return output_text
# print(output_text)
# print(i)

if __name__ == "__main__":

    init_sentence(25, 4)



























# from transformers import pipeline

# checkpoint = "/LaMini-GPT-124M" 

# model = pipeline('text-generation', model = checkpoint)

# instruction = 'Continue writing a paragraph about using the third person and describing human motion scenes that begins with the following sentence: someone dancing like a ballerina.'

# input_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

# generated_text = model(input_prompt, max_length=512, do_sample=True)[0]['generated_text']

# print("Response", generated_text)