import torch
import numpy as np
from dataset import dataset_TM_eval_token_think
import os
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
from options.option_train import get_args_parser
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == "__main__":
    args = get_args_parser()
    # load dataset for evaluation
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    args.dataname = 't2m'
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    # load dataset for training
    train_loader = dataset_TM_eval_token_think.DATALoader(args.dataname, "train", 1, w_vectorizer, unit_length=2**args.down_t, save_pkl=True)
                
