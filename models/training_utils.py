import torch
from torch.nn.utils import rnn
import re
# Training utils


def build_one_instance_t2m_qwen(tokenizer, captions, motion, motion_think):
    input_ids, target_ids = [], []
    bos = '<|im_start|>'
    bos_id = tokenizer(bos, add_special_tokens=False).input_ids
    input_ids.append(bos_id[0])
    target_ids.append(-100)  # do not perform loss regression on human prompt
    texts = ''
    prompt = "system\nYou are an assistant who helps users generate 3D human motion representations.\n\n"
    instruction = "### Instruction:\nThe users will describe a motion, your job is to generate a motion matching the following input human motion description. Show your reasoning inside <think>...</think> and output motion in <Motion>...</Motion> tags.<|im_end|>\n"
    input_text = '<|im_start|>user\n### Input:\n' + captions + '<|im_end|>\n<|im_start|>assistant\n'
    text = prompt + instruction + input_text
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt
    
    motion_think_ids = tokenizer(motion_think, add_special_tokens=False).input_ids
    input_ids += motion_think_ids
    target_ids += motion_think_ids

    pre = '<Motion>'
    pre_input_id = tokenizer(pre, add_special_tokens=False).input_ids
    text = '</Motion><|im_end|>'
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids

    input_ids += pre_input_id + motion.tolist() + one_input_id
    target_ids += pre_input_id + motion.tolist() + one_input_id
    return input_ids, target_ids

def build_one_instance_m2t_qwen(tokenizer, captions, motion, motion_think):
    motion = motion.astype(int)
    input_ids, target_ids = [], []
    bos = '<|im_start|>'
    bos_id = tokenizer(bos, add_special_tokens=False).input_ids
    input_ids.append(bos_id[0])
    target_ids.append(-100)  # do not perform loss regression on human prompt
    texts = ''
    prompt = "system\nYou are an assistant who helps users understand 3D human motion representations.\n\n"
    instruction = "### Instruction:\nThe users will provide the 3D human motion representations in <Motion>...</Motion> tags, your job is to generate the description matching the input motion. Show your reasoning inside <think>...</think> and output the description in <Answer>...</Answer> tags.<|im_end|>\n"
    input_text = '<|im_start|>user\n### Input:\n' + '<Motion>' + tokenizer.decode(motion) + '</Motion>' + '<|im_end|>\n<|im_start|>assistant\n'
    text = prompt + instruction + input_text
    texts += text
    one_input_id = tokenizer(text, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += [-100] * len(one_input_id)  # do not perform loss regression on human prompt
    
    answer = motion_think + '<Answer>' + captions + '</Answer><|im_end|>'
    answer_ids = tokenizer(answer, add_special_tokens=False).input_ids
    input_ids += answer_ids
    target_ids += answer_ids

    return input_ids, target_ids


def process_batch_motion_think(tokenizer, batch_of_captions, max_tgt_len, batch_of_motions, training_task, batch_of_motion_thinks):
    batch_input_ids, batch_target_ids = [], []
    for caption, motion, motion_think in zip(batch_of_captions, batch_of_motions, batch_of_motion_thinks):
        if training_task == 't2m':
            one_input_ids, one_target_ids = build_one_instance_t2m_qwen(tokenizer, caption, motion, motion_think)
        elif training_task == 'm2t':
            one_input_ids, one_target_ids = build_one_instance_m2t_qwen(tokenizer, caption, motion, motion_think)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))

    batch_input_ids = [seq.flip(0) for seq in batch_input_ids]
    batch_target_ids = [seq.flip(0) for seq in batch_target_ids]
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)

    input_ids = input_ids.flip(1)
    target_ids = target_ids.flip(1)

    input_ids = input_ids[:, -max_tgt_len:]
    target_ids = target_ids[:, -max_tgt_len:]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert input_ids.size() == target_ids.size()
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()