import torch
import pytorch_lightning as pl
import numpy as np
import os
import torch.nn.functional as F
from models.training_utils import *
import pickle
class MotionLLMLightning(pl.LightningModule):
    def __init__(self, args, w_vectorizer, eval_wrapper):
        super().__init__()
        self.save_hyperparameters(ignore=['w_vectorizer', 'eval_wrapper'])
        from models.mllm import MotionLLM 
        self.model = MotionLLM(args)
        self.model.net.eval()
        self.w_vectorizer = w_vectorizer
        self.eval_wrapper = eval_wrapper
        self.args = args
        self.best_fid = 1000
        self.best_val_loss = 1000
        self.best_val_acc = 0
        self.training_task = args.training_task
        self.train_outputs = []
        self.val_outputs = [] 

    def training_step(self, batch, batch_idx):
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name, motion_think = batch
        if self.current_epoch < 10:
            batch_task = 't2m'
        else:
            batch_task = np.random.choice(['t2m', 'm2t'])

    
        motion_tokens = []
        for i in range(motion.shape[0]):
            tokens = motion[i:i+1, :m_length[i]].squeeze(0).cpu().numpy() + (self.model.nb_text_tokens + 2)
            motion_tokens.append(tokens)
            
        input_ids, targets, attention_mask = process_batch_motion_think(
            tokenizer=self.model.tokenizer,
            batch_of_captions=caption,
            max_tgt_len=300,
            batch_of_motions=motion_tokens,
            training_task=batch_task,
            batch_of_motion_thinks=motion_think
        )

        input_ids = input_ids.to(self.device)
        targets = targets.to(self.device)
        attention_mask = attention_mask.to(self.device)


        outputs = self.model.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            labels=targets,
        )
        loss = outputs.loss

        if batch_task == 't2m':
            answer_start_ids = self.model.tokenizer("<Motion>", add_special_tokens=False).input_ids
            answer_end_ids = self.model.tokenizer("</Motion>", add_special_tokens=False).input_ids
        elif batch_task == 'm2t':
            answer_start_ids = self.model.tokenizer("<Answer>", add_special_tokens=False).input_ids
            answer_end_ids = self.model.tokenizer("</Answer>", add_special_tokens=False).input_ids
            
        answer_start_id = answer_start_ids[0]
        answer_end_id = answer_end_ids[0]

        chosen_tokens = torch.max(outputs.logits, dim=-1)[1][:, 1:-1]
        labels = targets[:, 2:]  

        gen_acc = (chosen_tokens.reshape(-1) == labels.reshape(-1)).to(torch.long)
        valid_mask = (labels != -100).reshape(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / (valid_mask.sum().item() + 1.0)

        cot_correct, cot_total = 0, 0
        answer_correct, answer_total = 0, 0
        cot_losses, answer_losses = [], []

        for i in range(labels.size(0)):
            label = labels[i]           # shape: [seq_len]
            pred = chosen_tokens[i]     # shape: [seq_len]
            logit = outputs.logits[i, 1:-1]  # shape: [seq_len, vocab_size]

            answer_start_pos = (label == answer_start_id).nonzero(as_tuple=False)
            answer_end_pos = (label == answer_end_id).nonzero(as_tuple=False)
            if answer_start_pos.numel() > 0 and answer_end_pos.numel() > 0:
                start_idx = answer_start_pos[0].item()
                end_idx = answer_end_pos[0].item() + 1
            else:
                start_idx = -1
                end_idx = -1

            valid_mask = (label != -100)
            answer_mask = torch.zeros_like(label, dtype=torch.bool)
            if 0 <= start_idx < end_idx <= label.size(0):
                answer_mask[start_idx:end_idx] = True
            cot_mask = valid_mask & (~answer_mask)

            # accuracy
            cot_correct += ((pred == label) & cot_mask).sum().item()
            cot_total += cot_mask.sum().item()
            answer_correct += ((pred == label) & answer_mask).sum().item()
            answer_total += answer_mask.sum().item()

            # loss
            if answer_mask.sum().item() > 0:
                ans_logits = logit[answer_mask]
                ans_labels = label[answer_mask]
                ans_loss = F.cross_entropy(ans_logits, ans_labels, reduction="mean")
                answer_losses.append(ans_loss)
            if cot_mask.sum().item() > 0:
                cot_logits = logit[cot_mask]
                cot_labels = label[cot_mask]
                cot_loss = F.cross_entropy(cot_logits, cot_labels, reduction="mean")
                cot_losses.append(cot_loss)

        if len(cot_losses) > 0:
            cot_loss_value = torch.stack(cot_losses).mean()
        else:
            cot_loss_value = torch.tensor(0.0, device=loss.device)
        if len(answer_losses) > 0:
            answer_loss_value = torch.stack(answer_losses).mean()
        else:
            answer_loss_value = torch.tensor(0.0, device=loss.device)

        cot_acc = cot_correct / (cot_total + 1e-8)
        answer_acc = answer_correct / (answer_total + 1e-8)
        
        lambda_answer = 0  # 可调节motion区间的权重
        final_loss = loss + lambda_answer * answer_loss_value
        output = {
            "loss": final_loss,
            "gen_acc": torch.tensor((cot_correct + answer_correct) / (cot_total + answer_total + 1e-8), device=loss.device),
            "cot_acc": torch.tensor(cot_acc, device=loss.device),
            "answer_acc": torch.tensor(answer_acc, device=loss.device),
            "cot_loss": cot_loss_value,
            "answer_loss": answer_loss_value,
        }

        if batch_idx % 200 == 0:
            if batch_idx == 0:
                print("motion_tokens", motion_tokens[0] -(self.model.nb_text_tokens + 2))
            if self.trainer.is_global_zero:
                print(f"Step {batch_idx} | loss: {output['loss'].item():.4f}, gen_acc: {output['gen_acc'].item():.4f}, cot_acc: {output['cot_acc'].item():.4f}, answer_acc: {output['answer_acc'].item():.4f}, cot_loss: {output['cot_loss'].item():.4f}, answer_loss: {output['answer_loss'].item():.4f}")

        self.train_outputs.append(output)
        return output

    
    def on_train_epoch_end(self):
        if len(self.train_outputs) == 0:
            return

        avg_loss = torch.stack([x['loss'] for x in self.train_outputs]).mean()
        avg_acc = torch.stack([x['gen_acc'] for x in self.train_outputs]).mean()
        avg_cot_acc = torch.stack([x['cot_acc'] for x in self.train_outputs]).mean()
        avg_answer_acc = torch.stack([x['answer_acc'] for x in self.train_outputs]).mean()
        avg_cot_loss = torch.stack([x['cot_loss'] for x in self.train_outputs]).mean()
        avg_answer_loss = torch.stack([x['answer_loss'] for x in self.train_outputs]).mean()

        self.log('train/loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/gen_acc', avg_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/cot_acc', avg_cot_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/answer_acc', avg_answer_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/cot_loss', avg_cot_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train/answer_loss', avg_answer_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.train_outputs.clear()

        if self.trainer.is_global_zero:
            ckpt_path = os.path.join(self.args.out_dir, f'motionllm_t2m_last')
            self.model.save_full_model(ckpt_path)
            if self.current_epoch >= self.args.epochs_start_val and self.current_epoch % self.args.epochs_val_interval == 0:
                ckpt_path = os.path.join(self.args.out_dir, f'motionllm_t2m_epoch{self.current_epoch}')
                self.model.save_full_model(ckpt_path)

    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        max_epochs = self.args.epochs

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=1e-5
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }