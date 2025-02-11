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
