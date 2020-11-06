import pytorch_lightning as pl

import argparse
from TransformerMarco import TransformerMarco
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.loggers import TensorBoardLogger
import os

from pytorch_lightning import seed_everything
seed_everything(42)

def main(hparams):
    model = TransformerMarco(hparams)

    loggers = []
    if hparams.use_wandb:
        wandb_logger = WandbLogger(project='long-marco', entity='usiir',
                            name=f'Albert-passage-{hparams.slurm_job_id}')
        wandb_logger.log_hyperparams(hparams)
        loggers.append(wandb_logger)
    if hparams.use_tensorboard:
        tb_logger = TensorBoardLogger("tb_logs", name=f"Longformer-docs",
                                    version=hparams.slurm_job_id)
        loggers.append(tb_logger)

    checkpoint_callback = ModelCheckpoint(
                filepath=os.path.join(os.getcwd(), 'checkpoints'),
                save_top_k=3,
                verbose=True,
                monitor='val_epoch_loss',
                mode='min',
                prefix=''
                )

    # This Trainer handles most of the stuff.
    # Enables distributed training with one line:
    # https://towardsdatascience.com/trivial-multi-node-training-with-pytorch-lightning-ff75dfb809bd
    trainer = pl.Trainer(
            gpus=hparams.gpus,
            num_nodes=hparams.num_nodes,
            distributed_backend=hparams.distributed_backend,
            # control the effective batch size with this param
            accumulate_grad_batches=hparams.trainer_batch_size,
            # Training will stop if max_steps or max_epochs have reached (earliest).
            max_epochs=hparams.epochs,
            max_steps=hparams.num_training_steps, 
            logger=loggers,
            checkpoint_callback=checkpoint_callback,
            # progress_bar_callback=False,
            # progress_bar_refresh_rate=0,
            # use_amp=True --> use 16bit precision
            # val_check_interval=0.25, # val 4 times during 1 train epoch
            val_check_interval=hparams.val_check_interval, # val every N steps
            # num_sanity_val_steps=5,
            # fast_dev_run=True
        )
    trainer.fit(model)


if __name__=='__main__':

    print(os.environ['SLURM_NODELIST'])
    parser = argparse.ArgumentParser(description='Transformer-MARCO')
    # MODEL SPECIFIC
    parser.add_argument("--max_seq_len", type=int, default=4096,
                        help="Maximum number of wordpieces of the sequence")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--num_warmup_steps", type=int, default=2500)
    parser.add_argument("--num_training_steps", type=int, default=120000)
    parser.add_argument("--val_check_interval", type=int, default=20000,
                        help='Run through dev set every N steps')
    parser.add_argument("--clf_dropout", type=float, default=-1.0,
                        help='Dropout for classifier. Set negative to use transformer dropout')
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Num subprocesses for DataLoader")

    # EXPERIMENT SPECIFIC
    parser.add_argument("--data_dir", type=str, default='~/long-marco/data',)
    parser.add_argument("--dataset", type=str, default='document',
                        help="`passage` or `document` re-ranking on MS MARCO")
    # effective batch size will be: 
    # trainer_batch_size * data_loader_bs
    parser.add_argument("--trainer_batch_size", type=int, default=5,
                        help='Batch size for Trainer. Accumulates grads every k batches')
    parser.add_argument("--data_loader_bs", type=int, default=1,
                        help='Batch size for DataLoader object')
    parser.add_argument("--val_data_loader_bs", type=int, default=0,
                        help='Batch size for validation data loader. If not specified,\
                        --data_loader_bs is used.')
    parser.add_argument("--use_10_percent_of_dev", type=int, default=1,
                        help='0 to use the full dev dataset, else to use 10% only')
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--slurm_job_id", type=int, default=1)
    parser.add_argument("--use_wandb", type=int, default=0, 
                        help='Use Weights&Biases (wandb) logger')
    parser.add_argument("--use_tensorboard", type=int, default=1,
                        help='Use TensorBoard logger (default in PL)')

    # Distributed training
    parser.add_argument("--gpus", type=int, default=1, help="Num of GPUs per node")
    parser.add_argument("--num_nodes", type=int, default=1, help="Num nodes allocated by SLURM")
    parser.add_argument("--distributed_backend", type=str, default='dp',
                        help="Use distributed backend: dp/ddp/ddp2")
    parser.add_argument("--model_name", type=str, default='allenai/longformer-base-4096',
                        help="Full name of: bert|albert|longformer")


    hparams = parser.parse_args()
    if hparams.val_data_loader_bs <= 0:
        hparams.val_data_loader_bs = hparams.data_loader_bs
    print(hparams)
    main(hparams)

