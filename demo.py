import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from models.mllm import MotionLLM
from options.option_train import get_args_parser
from utils.motion_utils import recover_from_ric, plot_3d_motion
from utils.paramUtil import t2m_kinematic_chain
import torch
import numpy as np
from tqdm import tqdm
import json

def t2m_demo():
    args = get_args_parser()
    args.bf16=True
    args.ddp = False
    args.add_tokens = False
    args.device = torch.device('cuda:0')
    args.llm_backbone = 'YOUR LLM PATH HERE'
    args.out_dir='./t2m_out/'
    os.makedirs(args.out_dir, exist_ok=True)
    model = MotionLLM(args)
    model.llm.eval()
    model.llm.cuda()
    
    caption="A person jumps up excitedly, throwing their arms over the head in celebration."

    motion, think_content = model.generate_motion_think(caption)
    motion = model.net.forward_decoder(motion[0])

    motion = model.denormalize(motion.detach().cpu().numpy())
    motion = recover_from_ric(torch.from_numpy(motion).float().cuda(), 22)
    joints_path = os.path.join(args.out_dir, f"t2m.npy")
    np.save(joints_path, motion.squeeze().detach().cpu().numpy())
    plot_3d_motion(args.out_dir + f"t2m.mp4", t2m_kinematic_chain, motion.squeeze().detach().cpu().numpy(), title=caption, think=think_content, fps=20, radius=4)

def m2t_demo():
    args = get_args_parser()
    args.bf16=True
    args.ddp = False
    args.add_tokens = False
    args.device = torch.device('cuda:0')
    args.llm_backbone = 'YOUR LLM PATH HERE'
    args.out_dir='./m2t_out/'
    os.makedirs(args.out_dir, exist_ok=True)
    model = MotionLLM(args)
    model.llm.eval()
    model.llm.cuda()
    joints_vecs_dir='./demo/joints_vecs/'
    captions_dict = {}

    npy_files = [f for f in os.listdir(joints_vecs_dir) if f.endswith('.npy')]
    
    for npy_file in tqdm(npy_files, desc="Processing motions"):
        file_key = os.path.splitext(npy_file)[0]

        motion_path = os.path.join(joints_vecs_dir, npy_file)
        motion = np.load(motion_path)
        motion = model.normalize(motion)
        motion = torch.from_numpy(motion).float().to(model.device).unsqueeze(0)
        motion_tokens = model.net.encode(motion).squeeze(0).cpu().numpy().astype(int) + model.nb_text_tokens - 514 -2
        caption = model.generate_caption_think(motion_tokens)
        captions_dict[file_key] = caption

    json_path = os.path.join(args.out_dir, 'captions_motion-unified.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(captions_dict, f, ensure_ascii=False, indent=2)
            
if __name__ == "__main__":
    m2t_demo()
