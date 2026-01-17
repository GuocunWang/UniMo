import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import dataset_TM_eval_token_think

class MotionDataModule(pl.LightningDataModule):
    def __init__(self, args, w_vectorizer):
        super().__init__()
        self.args = args
        self.w_vectorizer = w_vectorizer
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage=None):
        self.train_dataset = dataset_TM_eval_token_think.Text2MotionDataset(
            dataset_name=self.args.dataname,
            type="train",
            w_vectorizer=self.w_vectorizer,
            unit_length=4
        )


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=8,
            collate_fn=dataset_TM_eval_token_think.collate_fn,
            drop_last=True,
        )

