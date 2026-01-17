import torch
import numpy as np
from dataset import dataset_TM_code
import os
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from models.mllm import MotionLLM
from options.option_train import get_args_parser
from tqdm import tqdm
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    args = get_args_parser()
    args.bf16=True
    args.ddp = False
    args.add_tokens = False
    model = MotionLLM(args)
    model.eval()

    # load dataset for evaluation
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    args.dataname = 't2m'
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # load dataset for training
    train_loader = dataset_TM_code.DATALoader(args.dataname, "train", 1, w_vectorizer, unit_length=2**args.down_t)
    
    with torch.no_grad():
        for batch in tqdm(train_loader):
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch
            motion_tokens = []
            # encode the motion tokens
            for i in range(motion.shape[0]):
                tokens = model.net.encode(motion[i:i+1, :m_length[i], :].to(args.device)).squeeze(0)
                tokens = tokens.cpu().numpy()
                target_path = os.path.join("./dataset/HumanML3D/motionagent_vqvae", name[0] + '.npy')
                Path(target_path).parent.mkdir(parents=True, exist_ok=True)
                np.save(target_path, tokens)

                
