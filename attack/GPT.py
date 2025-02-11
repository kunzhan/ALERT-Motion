import openai
from zhipuai import ZhipuAI
import os
import re
import transformers
import torch
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.Timeout)), 
    wait=wait_random_exponential(multiplier=1, max=60), 
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def get_completion(prompt, top_p, num=1, model="gpt-3.5-turbo-instruct"):

    # openai.api_key = os.getenv('OPENAI_API_KEY')
    import zhipuai
    zhipuai.api_key = "41bf821f084345a6bb8194016ff8dc13.VdmNl6XV57RiIelv"
    model = "glm-4-plus"
    # openai.proxy = 'http://127.0.0.1:7890'#os.getenv('PROXY')
    # messages = [{"role": "user", "content": prompt}]
    client = zhipuai.ZhipuAI(api_key=zhipuai.api_key)
    if num==1:
        response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        top_p=top_p
        
        )
        return re.sub(r'[\'\n":]', '', response.choices[0].message.content), 0#response['usage']['total_tokens']
    else:
        responses = []
        for p in prompt:
            response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": p} 
            ],
            max_tokens=50,
            top_p=top_p
            )
            responses.append(response)
        stories = [""] * num
        words_to_remove = ["Prompt", "<prompt>", "</prompt>"]  # 你要去除的多个单词

        # 构建正则表达式模式，将多个单词以|连接
        pattern = r'\b(?:' + '|'.join(re.escape(word) for word in words_to_remove) + r')\b'

        for i, response in enumerate(responses):
            sentence = re.sub(r'[\'\n":]', '', response.choices[0].message.content)
            sentence = re.sub(pattern, '', sentence.lstrip())
            stories[i] = sentence.lstrip()
        return stories, 0#response['usage']['total_tokens']

    # return response.choices[0].message["content"]



if __name__ == "__main__":
    # sentence = "The man graceiously twirles and pirouetteds in his black suit, his movements captivated the eyes in awe and wonder of the on looker as the night bustle flicker in the neon lighting of"
    # I want you to work as a writing teacher. You need to study the given sentence, change only a few words from the original sentence according to the requirements and maintain the flow of the sentence. 
    # cand1 = "The man graceiously twirles and pirouetteds in his"
    # cand2 = "movements captivated the eyes in awe and wonder of the"
    prompt1 = "请总结下述论文综述，润色并总结，单词数不超过100，结果仍为英文："
    prompt2 = "以下是一篇学术论文中的一段内容，请将此部分翻译为英文，不要修改任何LaTeX命令，例如\section，\cite和方程式："
    prompt3 = "请重写这段论文综述，结果仍为英文"
    prompt4 = "请将下面的段落翻译为中文："
    prompt5 = "请将下面的方法描述翻译为英文，并根据此提出一个可以代表这个方法的英文名字，突出对抗性攻击任务："
    paragraphs = f'''
实现针对从文本到运动模型的对抗性提示生成将会遇到一些挑战。首先，与现有的图像分类对抗性攻击研究不同，我们的任务需要对文本添加扰动，由于文本的信息量相比于图像更少，即使只做很细微的扰动也更容易被人类发现。其次，不同于从文本到图像的对抗性攻击对对抗性提示词的语义要求不明显，人体运动生成模型的输入必须是与人体运动描述相关的提示词，在扰动时不考虑这种语义的约束将难以保证最终得到的结果仍满足人体运动描述。
    '''
    import os
 
    # os.environ["http_proxy"] = "http://localhost:7890"
    # os.environ["https_proxy"] = "http://localhost:7890"
    response = get_completion(prompt2+paragraphs, num=2, top_p=0.8)

    print(response)
    # pop_size = 15
    # cand1 = [['The', 'camera', 'pans', 'across', 'an', 'empty', 'field,', 'showing', 'trees', 'in', 'various', 'states', 'of', 'growth.'], ['The', 'camera', 'pans', 'across', 'an', 'uncultivated', 'field,', 'displaying', 'trees', 'in', 'various', 'states', 'of', 'development.'], ['The', 'camera', 'moves', 'across', 'an', 'unoccupied', 'field,', 'revealing', 'trees', 'in', 'different', 'stages', 'of', 'growth.'], ['The', 'camera', 'pans', 'across', 'an', 'empty', 'field,', 'showing', 'trees', 'in', 'various', 'states', 'of', 'growth.']]
    # cand2 = [['The', 'camera', 'pans', 'across', 'an', 'empty', 'field,', 'showing', 'trees', 'in', 'various', 'states', 'of', 'growth.'], ['The', 'camera', 'pans', 'across', 'an', 'uncultivated', 'field,', 'displaying', 'trees', 'in', 'various', 'states', 'of', 'development.'], ['The', 'camera', 'moves', 'across', 'an', 'unoccupied', 'field,', 'revealing', 'trees', 'in', 'different', 'stages', 'of', 'growth.'], ['The', 'camera', 'pans', 'across', 'an', 'empty', 'field,', 'showing', 'trees', 'in', 'various', 'states', 'of', 'growth.']]

    # #exchange two parts randomly
    # position = [[]] * pop_size
    # for i in range(pop_size):
    #     for j in range(len(cand1[i])):
    #         for k in range(len(cand2[i])):
    #             if cand1[i][j] == cand2[i][k] and abs(j-k)<4:
    #                 position[i].append([j, k])
    
    # next_pop = []
    # pop_index = 0
    # for pop_flag in position:
    #     index = np.random.choice(len(pop_flag), size=1)
    #     p = pop_flag[index.item()]
    #     p1_index, p2_index = p[0], p[1]
    #     # pop = cand1[pop_index][:p1_index] + cand2[pop_index][p2_index:]
    #     t1 = (cand1[pop_index][:p1_index]).join(' ')
    #     t2 = (cand2[pop_index][p2_index:]).join(' ')
    #     instruction = "My requirement is to construct a complete and flow sentence by combining two following clauses: '{}' and '{}'.".format(t1, t2)
    #     pop = generate(instruction)
    #     next_pop.append(pop)
    #     pop_index += 1