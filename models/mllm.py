from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn as nn
import torch
from models.training_utils import *
import numpy as np
import models.vqvae as vqvae
from typing import List, Union

class MotionLLM(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.llm_backbone)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        if self.args.bf16:
            self.llm = AutoModelForCausalLM.from_pretrained(self.args.llm_backbone, torch_dtype=torch.bfloat16)
        else:
            self.llm = AutoModelForCausalLM.from_pretrained(self.args.llm_backbone)
        self.nb_text_tokens = len(self.tokenizer)
        self.mean = np.load('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy')
        self.std = np.load('checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy')
        self.device = args.device
        self.training_task = None # t2m or m2t for training

        self.args.nb_joints = 22
        self.args.dataname = 't2m'
        self.args.vq_path = "ckpt/vqvae.pth"
        self.net = vqvae.HumanVQVAE(self.args, ## use args to define different parameters in different quantizers
                           self.args.nb_code,
                           self.args.code_dim,
                           self.args.output_emb_width,
                           self.args.down_t,
                           self.args.stride_t,
                           self.args.width,
                           self.args.depth,
                           self.args.dilation_growth_rate,
                           self.args.vq_act,
                           self.args.vq_norm)
        print ('loading vqvae from {}'.format(self.args.vq_path))
        ckpt = torch.load(self.args.vq_path, map_location='cpu')
        self.net.load_state_dict(ckpt['net'], strict=True)
        self.net.eval()
        if not self.args.ddp:
            self.net.to(self.device)
        for param in self.net.parameters():
            param.requires_grad = False

        if self.args.add_tokens:
            self.tokenizer.add_tokens(['<Motion>', '</Motion>'])
            self.motion_token_indices = np.arange(self.args.nb_code) 
            self.motion_token_indices = len(self.tokenizer) + self.motion_token_indices
            for i in range(self.args.nb_code):
                self.tokenizer.add_tokens([f'<Motion_{i}>'])
            self.tokenizer.add_tokens(['<think>', '</think>'])
            self.tokenizer.add_tokens(['<Answer>', '</Answer>'])
                
            self.llm.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)

        if not self.args.ddp:
            self.llm.to(self.device)
        self.llm.eval()
    
    
    def generate_motion_think(self, caption):
        prompt = "system\nYou are an assistant who helps users generate 3D human motion representations.\n\n"
        instruction = "### Instruction:\nThe users will describe a motion, your job is to generate a motion matching the following input human motion description. Show your reasoning inside <think>...</think> and output motion in <Motion>...</Motion> tags.<|im_end|>\n"
        input_text = '<|im_start|>user\n### Input:\n' + caption + '<|im_end|>\n<|im_start|>assistant\n<think>'
        prefix_text = prompt + instruction + input_text
        
        source_encoding = self.tokenizer(prefix_text,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)


        outputs = self.llm.generate(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            max_new_tokens=512)

        prompt_length = source_input_ids.shape[-1]

        new_tokens = outputs[:, prompt_length:]  # [batch_size, new_token_count]
        generated_texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        m = re.search(r"(.*?)</think>", generated_texts[0], flags=re.S)
        if m:
            think_content = m.group(1) 

        motion_tokens, cleaned_text = self.motion_string_to_token(generated_texts)
        
        return motion_tokens, think_content
    
    def generate_caption_think(self, motion):

        prompt = "system\nYou are an assistant who helps users understand 3D human motion representations.\n\n"
        instruction = "### Instruction:\nThe users will provide the 3D human motion representations in <Motion>...</Motion> tags, your job is to generate the description matching the input motion. Show your reasoning inside <think>...</think> and output the description in <Answer>...</Answer> tags.<|im_end|>\n"
        input_text = '<|im_start|>user\n### Input:\n' + '<Motion>' + self.tokenizer.decode(motion) + '</Motion>' + '<|im_end|>\n<|im_start|>assistant\n<think>'
        prefix_text = prompt + instruction + input_text
        
        source_encoding = self.tokenizer(prefix_text,
                                         return_attention_mask=True,
                                         add_special_tokens=True,
                                         return_tensors="pt")

        source_input_ids = source_encoding.input_ids.to(self.device)
        source_attention_mask = source_encoding.attention_mask.to(self.device)


        outputs = self.llm.generate(
            input_ids=source_input_ids,
            attention_mask=source_attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True,
            max_new_tokens=512)

        prompt_length = source_input_ids.shape[-1]

        new_tokens = outputs[:, prompt_length:]  # [batch_size, new_token_count]
        generated_texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        m = re.search(r"(.*?)</think>", generated_texts[0], flags=re.S)
        if m:
            think_content = m.group(1) 
            # print(think_content)
        caption = self.extract_answer_content(generated_texts[0])
        
        return caption, think_content
    
    def extract_answer_content(self, text):
        match = re.search(r"<Answer>(.*?)</Answer>", text, re.S)
        if match:
            return match.group(1)
        return ""
    
    def motion_string_to_token(self, motion_string: List[str]):
        motion_tokens = []
        output_string = []
        for i in range(len(motion_string)):
            prefix = '<Motion>'

            string= None
            string = self.get_middle_str_last(motion_string[i], prefix, '</Motion>')
            
            string_list = string.split('><')
            token_list = []
            for item in string_list[1:-1]:
                try:
                    token = int(item.split('_')[-1].replace('>', ''))
                    token_list.append(token)
                except ValueError:
                    pass
            if len(token_list) == 0:
                token_list = [0]
            token_list_padded = torch.tensor(token_list,
                                             dtype=int).to(self.device)
            motion_tokens.append(token_list_padded)
            output_string.append(motion_string[i].replace(
                string, '<Motion_Placeholder>'))

        return motion_tokens, output_string
    
    def get_middle_str_last(self, content, startStr, endStr):
        try:
            startIndex = content.rfind(startStr)
            endIndex = content.rfind(endStr)
            if startIndex == -1 or endIndex == -1 or endIndex <= startIndex:
                return '<Motion><Motion_0></Motion>'
            startIndex += len(startStr)
        except Exception as e:
            return '<Motion><Motion_0></Motion>'

        return '<Motion>' + content[startIndex:endIndex] + '</Motion>'
        
    def save_full_model(self, path):
        self.llm.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def denormalize(self, motion):
        return self.mean + motion * self.std

    def normalize(self, motion):
        return (motion - self.mean) / self.std
    
