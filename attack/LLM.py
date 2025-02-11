from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model



model_path = "1/InstructZero/WizardLM-13B-V1.2"
config = LlamaConfig.from_pretrained(model_path)
with init_empty_weights():
    model = LlamaForCausalLM._from_config(config) 
no_split_module_classes = LlamaForCausalLM._no_split_modules
device_map = infer_auto_device_map(model, max_memory = { 0: "0.1MIB", 1: "0.1MIB", 2: "20.0GIB", 3: "20.0GIB", 4: "20.0GIB", 5: "20.0GIB", 6: "20.0GIB", 7: "20.0GIB"}, no_split_module_classes=no_split_module_classes) #自动划分每个层的设备
# 可以尝试device_map为"balanced"是否成功，如果成功，无需手动设置device_map
# model = LlamaForCausalLM.from_pretrained("1/InstructZero/WizardLM-13B-V1.2", trust_remote_code=False, device_map="balanced")
load_checkpoint_in_model(model, model_path, device_map=device_map) #加载权重
model = dispatch_model(model,device_map=device_map) #并分配到具体的设备上