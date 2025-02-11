import copy
from nltk.tokenize import RegexpTokenizer, word_tokenize
from english import ENGLISH_FILTER_WORDS
import random
import sys,os
# from word2vec.word2vec_embed import Word2VecSubstitute
from transformers import pipeline
import shutil
import numpy as np
from compute_motion_sim import compute_motion_sim
import enchant
sys.path.append('target_model/mdm')# 当前工作目录
from motion_from_text import mdm_gen_motion_from_text
from sentence_transformers import SentenceTransformer, util


def is_english_word(word):
    d = enchant.Dict("en_US")
    # word可能是空字符串
    try:
        d.check(word)
    except ValueError:
        return True
    if d.check(word)==False and word in {',','.',':'}:
        return True
    return d.check(word)


class Error(Exception):
    """Base class for other exceptions"""
    pass

class WordNotInDictionaryException(Error):
    """Raised when the input value is too small"""
    pass

class attack():
    def __init__(self, tar_sent, device):
        #nothing to be initialized
        
        self.count = 0

        # tokenizer = RegexpTokenizer(r'\w+')
        tokens = word_tokenize(tar_sent)
        self.device = device

        #filter unimportant word

        target_word_ls = []
        for token in tokens:
            if token.lower() in ENGLISH_FILTER_WORDS:
                continue
            target_word_ls.append(token)
        self.target_sent_tokens = target_word_ls
        print("tar_sent_tokens: ", self.target_sent_tokens)

        # self.Word2vec = Word2VecSubstitute(tar_tokens=self.target_sent_tokens, device="cuda:6")

        print("initialize attack class.")

    
    def selectBug(self, x_prime, nums=1, if_initial=False):
        bugs = self.generateBugs(x_prime, nums, if_initial)
        target_num = random.randint(0, len(bugs)-1)
        bugs_ls = list(bugs.values())
        #randomly select a bug to return
        bug_choice = bugs_ls[target_num]
        return bug_choice
    

    # def replaceWithBug(self, x_prime, word_idx, bug):
    #     if bug == '':
    #         return x_prime[:word_idx] + x_prime[word_idx + 1:]
    #     else:
    #         return x_prime[:word_idx] + bug.split() + x_prime[word_idx + 1:]
        # return x_prime[:word_idx] + [bug] + x_prime[word_idx + 1:]


    def generateBugs(self, sentence, nums=1, if_initial=False):
        
        if if_initial:
            bugs = {"init": sentence}
            bugs["init"] = "My requirement is to generate {} sentences to change only a few random words, the original sentence: {}".format(nums, sentence)
            # bugs["insert"] = "Please insert a word in the following sentence for better flow: {}".format(sentence)
            # bugs["sub_W"] = "Please substitute a word in the following sentence for better flow: {}".format(sentence)
            # bugs["del_W"] = "Please delete a word in the following sentence for better flow: {}".format(sentence)
            # bugs["rev_S"] = "Please reverse the order of sentence expression while retaining the meaning of the sentence and rewrite the following sentence: {}".format(sentence)
        else:
            bugs = {"rewrite": sentence}

            # bugs["sub_W"] = "Please substitute a word in the following sentence for better flow: {}".format(sentence)
            # bugs["insert"] = "Please delete a word in the following sentence for better flow: {}".format(sentence)
            bugs["rewrite"] = f"""Please read the prompt following the <prompt> tag and rewrite it in a way that is different than the original. 
            You can add or remove portions.
            Replace words with synonyms and antonyms.
            Change the subject of the prompt.
            Reverse the order of the sentence.
            Only respond with a motion description. Do not include the <prompt> tag or anything before or after the prompt.
                 
            <prompt>
            {sentence}
            """
            # bugs["rewrite"] = "My requirement is to change only a few random words, the original sentence: {}".format(sentence)
            # bugs["rev_S"] = "My requirement is to reverse the order of the sentence, the original sentence: {}".format(sentence)

        return bugs

    # def bug_sub_tar_W(self, word):
    #     word_index = random.randint(0, len(self.target_sent_tokens) - 1)
    #     tar_word = self.target_sent_tokens[word_index]
    #     res = self.Word2vec.substitute(tar_word)
    #     if len(res) == 0:
    #         return word
    #     return res[0][0]

    def bug_sub_W(self, tokens, word, word_idx):
        try:
            # res = self.Word2vec.substitute(word)
            unmasker = pipeline('fill-mask', model='/xlm-roberta-base', device=self.device)
            tokens_ = copy.deepcopy(tokens)
            tokens_[word_idx] = '<mask>'
            sentence = " ".join(tokens_)
            res_dict_list = unmasker(sentence, top_k=16)
            candidates = [res_dict['token_str'] for res_dict in res_dict_list]
            # probs = []
            scores = np.array([res_dict['score'] for res_dict in res_dict_list])
            # for i, res_dict in enumerate(res_dict_list):
            scores -= np.max(scores)
            probs = np.exp(scores) / np.sum(np.exp(scores))
            res = np.random.choice(np.arange(len(res_dict_list)), size=1, p=probs)
            i = 0
            while res.item() == word_idx or not is_english_word(candidates[res.item()]):
                res = np.random.choice(np.arange(len(res_dict_list)), size=1, p=probs)
                if i > 15:
                    return word
                i = i + 1

            # if len(res) == 0:
            #     return word
            return candidates[res.item()]
        except WordNotInDictionaryException:
            return word

    def bug_insert_W(self, tokens, word, word_idx):
        try:
            # res = self.Word2vec.substitute(word)
            unmasker = pipeline('fill-mask', model='/xlm-roberta-base', device=self.device)
            tokens_ = copy.deepcopy(tokens)
            tokens_ = tokens_[:word_idx+1] + ['<mask>'] + tokens_[word_idx+1:]
            sentence = " ".join(tokens_)
            res_dict_list = unmasker(sentence, top_k=6)
            candidates = [res_dict['token_str'] for res_dict in res_dict_list]
            # probs = []
            scores = np.array([res_dict['score'] for res_dict in res_dict_list])
            # for i, res_dict in enumerate(res_dict_list):
            scores -= np.max(scores)
            probs = np.exp(scores) / np.sum(np.exp(scores))
            res = np.random.choice(np.arange(len(res_dict_list)), size=1, p=probs)
            i = 0
            while not is_english_word(candidates[res.item()]):
                res = np.random.choice(np.arange(len(res_dict_list)), size=1, p=probs)
                if i > 15:
                    return word
                i = i + 1

            # if len(res) == 0:
            #     return word
            return word + ' ' + candidates[res.item()]
        except WordNotInDictionaryException:
            return word

    def bug_delete_W(self):

        res = ""
        return res

    # def bug_swap(self, word):
    #     if len(word) <= 4:
    #         return word
    #     res = word
    #     points = random.sample(range(1, len(word) - 1), 2)
    #     a = points[0]
    #     b = points[1]

    #     res = list(res)
    #     w = res[a]
    #     res[a] = res[b]
    #     res[b] = w
    #     res = ''.join(res)
    #     return res

    # def bug_random_sub(self, word):
    #     res = word
    #     point = random.randint(0, len(word)-1)

    #     choices = "qwertyuiopasdfghjklzxcvbnm"
        
    #     subbed_choice = choices[random.randint(0, len(list(choices))-1)]
    #     res = list(res)
    #     res[point] = subbed_choice
    #     res = ''.join(res)
    #     return res
    
    # def bug_convert_to_leet(self, word):
    #     # Dictionary that maps each letter to its leet speak equivalent.
    #     leet_dict = {
    #         'a': '4',
    #         'b': '8',
    #         'e': '3',
    #         'g': '6',
    #         'l': '1',
    #         'o': '0',
    #         's': '5',
    #         't': '7'
    #     }
        
    #     # Replace each letter in the text with its leet speak equivalent.
    #     res = ''.join(leet_dict.get(c.lower(), c) for c in word)
        
    #     return res


    # def bug_sub_C(self, word):
    #     res = word
    #     key_neighbors = self.get_key_neighbors()
    #     point = random.randint(0, len(word) - 1)

    #     if word[point] not in key_neighbors:
    #         return word
    #     choices = key_neighbors[word[point]]
    #     subbed_choice = choices[random.randint(0, len(choices) - 1)]
    #     res = list(res)
    #     res[point] = subbed_choice
    #     res = ''.join(res)

    #     return res

    # def get_key_neighbors(self):
    #     ## TODO: support other language here
    #     # By keyboard proximity
    #     neighbors = {
    #         "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
    #         "i": "uojkl", "o": "ipkl", "p": "ol",
    #         "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
    #         "j": "yuihknm", "k": "uiojlm", "l": "opk",
    #         "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
    #     }
    #     # By visual proximity
    #     neighbors['i'] += '1'
    #     neighbors['l'] += '1'
    #     neighbors['z'] += '2'
    #     neighbors['e'] += '3'
    #     neighbors['a'] += '4'
    #     neighbors['s'] += '5'
    #     neighbors['g'] += '6'
    #     neighbors['b'] += '8'
    #     neighbors['g'] += '9'
    #     neighbors['q'] += '9'
    #     neighbors['o'] += '0'

    #     return neighbors

def sort_words_by_importance(save_dir, ori_sent, tar_motion_path):
    tokens = word_tokenize(ori_sent)
    sim_ls = []
    sentences = []
    print("----------------------------------------------------------------\n")
    for i in range(min(len(tokens),20)):
        new_tokens = tokens[:i] + tokens[i+1:]
        x_prime_sent = " ".join(new_tokens)
        
        x_motion_path = save_dir + "gen.npy"

        mdm_gen_motion_from_text(x_prime_sent, x_motion_path)

        similarity = compute_motion_sim(x_motion_path, tar_motion_path)
        
        print(f"x_prime_sent: {x_prime_sent}, similarity: {similarity.item()}")
        sim_ls.append(similarity.item())
        sentences.append(x_prime_sent)
    print("----------------------------------------------------------------\n")
    sim_arr = np.array(sim_ls)   
    scores_logits = np.exp(sim_arr - sim_arr.max()) 
    sim_probs = scores_logits / scores_logits.sum()
    
    return sim_probs


def sentence_distance(sentences1, sentences2):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Two lists of sentences
    # sentences1 = 'The person walked forward and is picking up his toolbox.'

    # sentences2 = 'The camera pans across an empty field, showing nothing happening in particular as a person stumbles around.'

    #Compute embedding for both lists
    embeddings1 = model.encode([sentences1], convert_to_tensor=True)
    embeddings2 = model.encode([sentences2], convert_to_tensor=True)

    #Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    return 1-cosine_scores.item()

# if __name__ == "__main__":
#     sentence_distance()