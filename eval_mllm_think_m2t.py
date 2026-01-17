from options.option_train import get_args_parser
from models.mllm import MotionLLM
import torch
from dataset import dataset_TM_eval_think
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
import numpy as np
import os
from tqdm import tqdm
import json
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def eval_m2t():
    args = get_args_parser()
    args.bf16=True
    args.ddp = False
    args.add_tokens = False
    args.device = torch.device('cuda:0')
    args.llm_backbone = 'YOUR LLM PATH HERE'
    model = MotionLLM(args)
    model.llm.to(args.device)
    model.net.to(args.device)
    model.llm.eval()
    model.net.eval()
    
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    args.dataname = 't2m'
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    test_loader = dataset_TM_eval_think.DATALoader(args.dataname, "test", 32, w_vectorizer, unit_length=2**args.down_t) # batch size should be fixed to 32

    repeat_time = 20
    os.makedirs("./m2t_results", exist_ok=True)
    for i in range(repeat_time):
        json_path = f'./m2t_results/m2t_results_{i}.json'
        batch_idx = 0
        results = {} 
        for batch in tqdm(test_loader):
            batch_idx += 1
            word_embeddings, pos_one_hots, caption, sent_len, pose, m_length, token, name, motion_think = batch
            bs, seq = pose.shape[:2]
            for k in range(bs):
                motion_tokens = model.net.encode(
                    pose[k:k+1, :m_length[k], :].to(args.device)
                ).squeeze(0)
                motion_tokens_llm = motion_tokens.cpu().numpy().astype(int) + model.nb_text_tokens - 514 -2
                motion_caption, think_content = model.generate_caption_think(motion_tokens_llm)
                if batch_idx == 1 and k == 0:
                    print("think_content", think_content)
                    print("motion_tokens", motion_tokens)
                    print("motion_caption", motion_caption)
                key = name[k]
                results[key] = {
                    "motion_caption": motion_caption,
                    "caption": caption[k]
                }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
    print(args.llm_backbone)

if __name__ == "__main__":
    eval_m2t()