import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"
from tqdm import tqdm
import json
import argparse
import glob
from pathlib import Path
from statistics import mean

import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from pycocoevalcap.cider.cider import Cider


nltk.download('punkt_tab')

def load_preds_refs(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    hyps, refs = [], []
    for _, v in data.items():
        hyps.append(v["motion_caption"])
        refs.append(v["caption_list"])
    return hyps, refs

def tokenize(sent):
    return [w.lower() for w in nltk.word_tokenize(sent)]


def compute_bleu(hyps, refs, weights):
    refs_tok = [[tokenize(r) for r in ref_list] for ref_list in refs]
    hyps_tok = [tokenize(h) for h in hyps]
    smooth_fn = SmoothingFunction().method4
    return corpus_bleu(refs_tok, hyps_tok, weights=weights, smoothing_function=smooth_fn)


def compute_rouge_l(hyps, refs):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    f_scores = []
    for hyp, ref_list in zip(hyps, refs):
        best_f = max(scorer.score(ref, hyp)['rougeL'].fmeasure for ref in ref_list)
        f_scores.append(best_f)
    return mean(f_scores)


def compute_cider(hyps, refs):
    gts, res = {}, {}
    for i, (h, r) in enumerate(zip(hyps, refs)):
        gts[i] = r
        res[i] = [h]
    cider_scorer = Cider()
    score, _ = cider_scorer.compute_score(gts, res)
    return score


def process_one(json_path):
    hyps, refs = load_preds_refs(json_path)
    bleu1 = compute_bleu(hyps, refs, weights=(1, 0, 0, 0))
    bleu4 = compute_bleu(hyps, refs, weights=(0.25, 0.25, 0.25, 0.25))
    rouge_l = compute_rouge_l(hyps, refs)
    cider = compute_cider(hyps, refs)

    P, R, F1 = bert_score(hyps, refs, lang="en", rescale_with_baseline=True)
    bert_f1 = F1.mean().item()

    return dict(
        bleu1=bleu1,
        bleu4=bleu4,
        rouge_l=rouge_l,
        cider=cider,
        bert_f1=bert_f1,
    )


def main(args):
    pattern = str(args.dir / "m2t_results_*_match.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files match pattern: {pattern}")
        return

    metrics = []
    for fp in tqdm(files):
        print(f"Evaluating: {os.path.basename(fp)}")
        metrics.append(process_one(fp))

    avg = {k: mean(m[k] for m in metrics) for k in metrics[0]}
    print("\n=====  Average over all files  =====")
    print(f"BLEU-1 : {avg['bleu1']:.4f}")
    print(f"BLEU-4 : {avg['bleu4']:.4f}")
    print(f"ROUGE-L: {avg['rouge_l']:.4f}")
    print(f"CIDEr  : {avg['cider']:.4f}")
    print(f"BERTScore-F1: {avg['bert_f1']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluate motion captions with multiple references.")
    parser.add_argument("--dir", type=Path,
                        default="./m2t_results",
                        help="Directory containing m2t_results_*_match.json files")
    main(parser.parse_args())