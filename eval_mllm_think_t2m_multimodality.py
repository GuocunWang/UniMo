from options.option_train import get_args_parser
from models.mllm import MotionLLM
import torch
from utils.evaluation_think_multimodality import evaluation_test_think_multimodality
from dataset import dataset_TM_eval_think
from utils.word_vectorizer import WordVectorizer
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.get_eval_option import get_opt
import numpy as np
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def eval_t2m_multimodality():
    args = get_args_parser()
    args.bf16=True
    args.ddp = False
    args.add_tokens = False
    args.device = torch.device('cuda:0')
    args.llm_backbone = 'YOUR LLM PATH HERE'
    model = MotionLLM(args)
    print(args.llm_backbone)
    model.llm.to(args.device)
    model.net.to(args.device)
    model.llm.eval()
    model.net.eval()
    
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    args.dataname = 't2m'
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    test_loader = dataset_TM_eval_think.DATALoader(args.dataname, "test", 1, w_vectorizer, unit_length=2**args.down_t) # batch size should be fixed to 1

    multi = []
    repeat_time = 20
    for i in range(repeat_time):
        best_multi = evaluation_test_think_multimodality(args.out_dir, test_loader, model, eval_wrapper=eval_wrapper)
        multi.append(best_multi)


    print('final result:')
    print('multi: ', sum(multi)/repeat_time)

    multi = np.array(multi)
    msg_final = f"Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
    print(msg_final)
    print(args.llm_backbone)

if __name__ == "__main__":
    eval_t2m_multimodality()