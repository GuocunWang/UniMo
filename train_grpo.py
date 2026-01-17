import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import re
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
import json
import numpy as np
import torch.nn.functional as F
import clip
from options.option_train import get_args_parser
import torch
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
import models.vqvae as vqvae
import torch.distributed as dist
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

model_name = "YOUR FIRST STAGE LLM HERE"
 
output_dir="./experiments/UniMo-GRPO"
run_name="UniMo-GRPO"

local_rank = int(os.environ.get('LOCAL_RANK', -1))
if local_rank != -1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
t2m_prompt = "You are an assistant who helps users generate 3D human motion representations.\n\n"
t2m_instruction = "### Instruction:\nThe users will describe a motion, your job is to generate a motion matching the following input human motion description. Show your reasoning inside <think>...</think> and output motion in <Motion>...</Motion> tags."
    
T2M_SYSTEM_PROMPT = t2m_prompt + t2m_instruction

m2t_prompt = "You are an assistant who helps users understand 3D human motion representations.\n\n"
m2t_instruction = "### Instruction:\nThe users will provide the 3D human motion representations in <Motion>...</Motion> tags, your job is to generate the description matching the input motion. Show your reasoning inside <think>...</think> and output the description in <Answer>...</Answer> tags.<|im_end|>"
    
M2T_SYSTEM_PROMPT = m2t_prompt + m2t_instruction

args = get_args_parser()
args.nb_joints = 22
args.dataname = 't2m'
args.vq_path = "ckpt/vqvae.pth"
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        args.vq_act,
                        args.vq_norm)
print ('loading vqvae from {}'.format(args.vq_path))
ckpt = torch.load(args.vq_path, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
for param in net.parameters():
    param.requires_grad = False

net.cuda(local_rank)

net.eval()

w_vectorizer = WordVectorizer('./glove', 'our_vab')
args.dataname = 't2m'
dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
wrapper_opt = get_opt(dataset_opt_path, f'cuda:{local_rank}')
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
clip_model, _ = clip.load("ViT-B/32", device=f'cuda:{local_rank}')
clip_model.eval()

def get_text_embedding(tokens, w_vectorizer):
    max_text_len = 20
    if len(tokens) < max_text_len:
        # pad with "unk"
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
        tokens = tokens + ['unk/OTHER'] * (max_text_len + 2 - sent_len)
    else:
        # crop
        tokens = tokens[:max_text_len]
        tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        sent_len = len(tokens)
    pos_one_hots = []
    word_embeddings = []
    for token in tokens:
        word_emb, pos_oh = w_vectorizer[token]
        pos_one_hots.append(pos_oh[None, :])
        word_embeddings.append(word_emb[None, :])
    pos_one_hots = np.concatenate(pos_one_hots, axis=0)
    word_embeddings = np.concatenate(word_embeddings, axis=0)
    return word_embeddings, pos_one_hots, sent_len

def extract_motion_tokens(text: str) -> torch.Tensor:
    """
    Extracts Motion token IDs from the provided text inside <Motion> tags.
    Returns a NumPy array of integers corresponding to the Motion_<number> format.
    Skips invalid formats and returns only valid tokens.

    :param text: The text from which to extract Motion tokens.
    :return: A NumPy array of extracted Motion IDs.
    """
    answer = text.split("<Motion>")[-1]
    answer = answer.split("</Motion>")[0]
    motion_pattern = r"<Motion_(\d+)>"
    
    motion_tokens = re.findall(motion_pattern, answer)
    
    motion_ids = [int(token) for token in motion_tokens]

    return torch.tensor(motion_ids)

def extract_answer_content(text: str) -> str:
    match = re.search(r"<Answer>(.*?)</Answer>", text, re.S)
    if match:
        return match.group(1)
    return ""

def get_t2m_dataset(split="train") -> Dataset:
    with open('./motion_train.json', 'r') as f:
        data_json = json.load(f)
    
    prompts = []
    motions = []
    captions = []
    tokens = []
    f_tags = []
    to_tags = []
    t2m_tags = []
    for entry in data_json:
        t2m_tag = np.random.choice(['t2m', 'm2t'])
        if t2m_tag == 't2m':
            prompts.append([
                {'role': 'system', 'content': T2M_SYSTEM_PROMPT},
                {'role': 'user', 'content': '### Input:\n' + entry['caption']}
            ])
        else:
            prompts.append([
                {'role': 'system', 'content': M2T_SYSTEM_PROMPT},
                {'role': 'user', 'content': '### Input:\n' + entry['motion']}
            ])
        motions.append(entry['motion'])
        captions.append(entry['caption'])
        tokens.append(entry['tokens'])
        f_tags.append(entry['f_tag'])
        to_tags.append(entry['to_tag'])
        t2m_tags.append(t2m_tag)
    
    return Dataset.from_dict({
        'prompt': prompts,
        'motion': motions,
        'caption': captions,
        'tokens': tokens,
        'f_tag': f_tags,
        'to_tag': to_tags,
        't2m_tag': t2m_tags
    })

@torch.no_grad()  
def similarity_reward_func(completions, motion, caption, tokens, f_tag, to_tag, t2m_tag, net, w_vectorizer, clip_model, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]

    extracted_motion_tokens_gt = [extract_motion_tokens(a) for a in motion]

    if t2m_tag[0] == 't2m':
        extracted_motion_tokens = [extract_motion_tokens(r) for r in responses]
        pred_pose_scores = []
        
        for i in range(len(extracted_motion_tokens)):
            try:
                pred_pose = net.forward_decoder(extracted_motion_tokens[i].cuda(local_rank).long())
            except:
                index_motion = torch.ones(1,1).cuda().long()
                pred_pose = net.forward_decoder(index_motion)
                
            pred_pose_gt = net.forward_decoder(extracted_motion_tokens_gt[i].cuda(local_rank).long())

            word_embeddings, pos_one_hots, sent_len = get_text_embedding(tokens[i], w_vectorizer)

            word_embeddings = torch.tensor(word_embeddings).to(pred_pose.device)  
            pos_one_hots = torch.tensor(pos_one_hots).to(pred_pose.device)  
            sent_len = torch.tensor([sent_len]).long().to(pred_pose.device)
            

            word_embeddings = word_embeddings.unsqueeze(0)  
            pos_one_hots = pos_one_hots.unsqueeze(0)  

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose, torch.tensor([pred_pose.shape[1]]).long())
            et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_gt, torch.tensor([pred_pose_gt.shape[1]]).long())
            

            motion_cos_sim = F.cosine_similarity(em_pred, em, dim=1).item()  
            motion_text_cos_sim = F.cosine_similarity(em_pred, et, dim=1).item()  

            pred_pose_scores.append(motion_cos_sim + motion_text_cos_sim)
        return pred_pose_scores
    
    else:
        extracted_answers = [extract_answer_content(r) for r in responses]
        pred_answer_scores = []
        
        text_inputs = [caption[0]] + extracted_answers

        text_clip = clip.tokenize(text_inputs, truncate=True).to(f'cuda:{local_rank}')


        text_clip_emb = clip_model.encode_text(text_clip) 
        caption_emb = text_clip_emb[0]
        for i in range(len(extracted_answers)):               
            answer_emb = text_clip_emb[i+1] 
            text_cos_sim = F.cosine_similarity(caption_emb, answer_emb, dim=-1).item() 
            pred_answer_scores.append(text_cos_sim*2)
        return pred_answer_scores
    

 


def strict_format_reward_func(completions, t2m_tag, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    if t2m_tag[0] == 't2m':
        pattern = r"<think>.*?</think><Motion>(<Motion_\d+>)+</Motion>"
    else:
        pattern = r"<think>.*?</think><Answer>.*?</Answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [1.0 if match else 0.0 for match in matches]


 

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    shuffle_dataset=True,
    learning_rate=5e-5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=10,
    bf16=True,
    steps_per_generation=1,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_generations=8,
    max_prompt_length=256,
    max_completion_length=400,
    num_train_epochs=2,
    save_steps=1000,
    max_grad_norm=0.1,
    epsilon=0.2,
    beta=0.001,
    log_on_each_node=False,
    scale_rewards=False,
    use_vllm=True,
    vllm_gpu_memory_utilization=.2,
    vllm_tensor_parallel_size=1,
    vllm_mode="colocate",
    report_to="tensorboard"
)
 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None
).cuda()
 
model.train()
 
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
 
dataset = get_t2m_dataset()
 
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        strict_format_reward_func,
        lambda completions, motion, caption, tokens, f_tag, to_tag, t2m_tag,  **kwargs: similarity_reward_func(
            completions, motion, caption, tokens, f_tag, to_tag, t2m_tag, net, w_vectorizer, clip_model, **kwargs
        )
    ],
    args=training_args,
    train_dataset=dataset,
)

 
trainer.train()
 

# deepspeed --num_gpus=4 train_grpo.py --deepspeed_config ds_config.json