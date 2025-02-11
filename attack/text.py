from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
import torch,sys
import os,shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from compute_motion_sim import compute_motion_sim
sys.path.append('target_model/mdm')
from motion_from_text import mdm_gen_motion_from_text


class MotionScoresTokenLogitsProcessor(LogitsProcessor):
    """
    综合考虑token对文本流畅和生成运动与目标相似度的影响
    """
    def __init__(self, ori_sentence: str, input_len: int, top_k: int,
                tar_motion_path: str, tar_sentence: str, intem_motion_path: str,
                best_motion_path: str, log: str):
        """
        :param  ori_sentence: 原始的句子
                input_len: 原始句子的长度
        """
        self.ori_sentence = ori_sentence
        self.input_len = input_len
        self.top_k = top_k
        self.target_motion_path = tar_motion_path
        self.target_sent = tar_sentence
        # self.log_save_path = log_save_path
        self.intermediate_path = intem_motion_path
        self.best_motion_path = best_motion_path
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        itr = input_ids.shape[1] - self.input_len
        log.write("------------- iteration:" + str(itr) + " ---------------\n")
        
        if input_ids.shape[1] == self.input_len:
            generated_words = ""
        else:
            generated_words = tokenizer.decode(input_ids[0,input_len:])
        sorted_scores, ids = torch.sort(scores, descending=True)
        next_words = tokenizer.convert_ids_to_tokens(ids[0,:top_k])
        x_motion_path = self.intermediate_path + "gen.npy"
        sim_score_ls = []
        new_scores = torch.full(scores.shape, -float("Inf"))
        for i,next_word in enumerate(next_words):
            if list(next_word)[0] == 'Ġ':
                sentence = generated_words + ' ' + ''.join(list(next_word)[1:])
            else:
                sentence = generated_words + next_word
            mdm_gen_motion_from_text(sentence, x_motion_path)
            similarity = compute_motion_sim(x_motion_path, self.target_motion_path)
            sim_score_ls.append(similarity.item())
            print(f"x_prime_sent: {sentence}, similarity: {similarity.item()}")
            log.write(str(similarity.item()) + " " + sentence + "\n")

        sim_score = torch.FloatTensor(sim_score_ls)
        sim_z_score = (sim_score-sim_score.mean())/sim_score.std()
        for j in range(len(next_words)):
            # scores[:,ids[0,j]] = scores[:,ids[0,j]]+sim_z_score[j]
            new_scores[:,ids[0,j]] = sim_z_score[j]
        
        # global best_score
        # if sim_score.max() > best_score:
        best_score = sim_score.max()
        #store the current best motion
        sentence = generated_words + next_words[sim_score.argmax()].replace('Ġ', ' ').replace('$', '\'')
        
        x_motion_path_npy = self.intermediate_path + "gen.npy"
        x_motion_path_mp4 = self.intermediate_path + "gen.mp4"

        mdm_gen_motion_from_text(sentence, x_motion_path_npy)

        best_ori_path_npy = self.best_motion_path + "itr_" + str(itr) + "_score_" + str(best_score) + ".npy"
        best_ori_path_mp4 = self.best_motion_path + "itr_" + str(itr) + "_score_" + str(best_score) + ".mp4"
        
        shutil.copy(x_motion_path_npy, best_ori_path_npy)
        shutil.copy(x_motion_path_mp4, best_ori_path_mp4)

        log.write("new best adv: " +  str(sim_score.max()) + " " + sentence + "\n")
        log.flush()

        return new_scores
# sim_score = torch.zeros(5)
# value1, value2 = sim_score.max(), sim_score.argmax()
# next_word = 'Ġperson'
# ''.join(c=' ' if c=='Ġ' for c in list(next_word))

global best_score 
best_score = float("-inf")


top_k = 8
tar_motion_path = "/attack/target.npy"
intem_motion_path = '/attack/intermediate_motion_path_00/'
best_motion_path = '/attack/best_motion_path_00/'
log_save_path = 'run_log_00.txt'

device  = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("/LaMini-GPT-1.5B")
model = AutoModelForCausalLM.from_pretrained("/LaMini-GPT-1.5B", trust_remote_code=False)

ori_sentence = 'Someone dancing like a ballerina.'
tar_sentence = 'the person walked forward and is picking up his toolbox.'

# instruction = 'Continue writing a paragraph about using the third person and describing human motion scenes that begins with the following sentence: {}'.format(ori_sentence)
instruction = 'Rephrase the following sentence about using the third person and describing human motion scenes to two sentences, to avoid repetition, while keeping its meaning:{}'.format(ori_sentence)
input_prompt = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:".format(instruction)


input_ids = tokenizer.encode(input_prompt, return_tensors="pt")
# with torch.no_grad():
_, input_len = input_ids.shape

# tmp = model.forward(input_ids)
# a, idx1 = torch.sort(tmp.logits[:,-1,:], descending=True)#descending为alse，升序，为True，降序
# idx = idx1[0,:8]
# next_words = tokenizer.convert_ids_to_tokens(idx)
logits_processor = LogitsProcessorList()


with open(log_save_path, 'w') as log:
    if log is not None:
        log.write('target phrase: ' + tar_sentence + '\n')
        log.write('origin phrase:'+ ori_sentence + '\n')
    logits_processor.append(MotionScoresTokenLogitsProcessor(ori_sentence, input_len, top_k, tar_motion_path, tar_sentence, intem_motion_path, best_motion_path, log))
    output = model.generate(input_ids, max_length=24+input_len, logits_processor=logits_processor)# , temperature=0.7

output_text = tokenizer.decode(output[0, input_len:], skip_special_tokens=True)
print(output_text)



























# from transformers import pipeline

# checkpoint = "/LaMini-GPT-124M" 

# model = pipeline('text-generation', model = checkpoint)

# instruction = 'Continue writing a paragraph about using the third person and describing human motion scenes that begins with the following sentence: someone dancing like a ballerina.'

# input_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:"

# generated_text = model(input_prompt, max_length=512, do_sample=True)[0]['generated_text']

# print("Response", generated_text)