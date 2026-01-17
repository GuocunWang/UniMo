import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.word_vectorizer import WordVectorizer
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from options.option_train import get_args_parser

from model_lightning_think import MotionLLMLightning
from datamodule_think import MotionDataModule
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if __name__ == "__main__":
    args = get_args_parser()
    args.bf16=False
    args.ddp = True
    args.add_tokens = True
    args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    logger = TensorBoardLogger(save_dir=args.out_dir, name="lightning_logs")
    csv_logger = CSVLogger(save_dir=args.out_dir, name="csv_logs")

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    args.dataname = 't2m'
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    wrapper_opt = get_opt(dataset_opt_path, args.device)
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    data_module = MotionDataModule(args, w_vectorizer)
    model = MotionLLMLightning(args, w_vectorizer, eval_wrapper)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=[0,1,2,3],
        strategy="ddp_find_unused_parameters_true",
        accumulate_grad_batches=2,
        logger=[logger, csv_logger],
        enable_progress_bar=True,
        enable_checkpointing=False,
    )

    trainer.fit(model, datamodule=data_module)
