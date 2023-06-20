# conda codebert 
from transformers import RobertaTokenizer
from transformers import T5EncoderModel
import json 
import torch
from tqdm import tqdm 
import os 
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
device = torch.device("cpu")

path = "/zju_yetong/yetong_personal/cache/huggingface/hub/models--Salesforce--codet5-base/snapshots/4078456db09ba972a3532827a0b5df4da172323c/"
tokenizer = RobertaTokenizer.from_pretrained(path)
model = T5EncoderModel.from_pretrained(path)
model.to(device)

input = torch.randint(low=0,high=1000, size=(32,400))
input = input.to(device)
for item in tqdm(range(100)):
    hidden = model(input_ids=input).last_hidden_state
print(hidden)
print(hidden.size())
