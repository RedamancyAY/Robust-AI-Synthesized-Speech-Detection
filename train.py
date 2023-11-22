# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import argparse
import os
import random
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.model_summary import summarize
# -

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

from myutils.tools import to_list
from myutils.torch.lightning.callbacks import (
    ACC_Callback,
    APCallback,
    AUC_Callback,
    Color_progress_bar,
    EER_Callback,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from rich.console import Console

from config import get_cfg_defaults
from data.make_dataset import make_data
from models import (
    AASIST_lit,
    AudioClip_lit,
    AudioModel_lit,
    LCNN_lit,
    LibriSeVoc_lit,
    RawNet2_lit,
    Wav2Clip_lit,
    Wav2Vec2_lit,
    WaveLM_lit,
)
from utils.tools import backup_logger_file, get_ckpt_path

ROOT_DIR = "xxxx"


# + editable=true slideshow={"slide_type": ""}
def make_model(cfg_file, cfg):
    if cfg_file.startswith("LCNN/"):
        model = LCNN_lit()
    elif cfg_file.startswith("RawNet2/"):
        model = RawNet2_lit()
    elif cfg_file.startswith("WaveLM/"):
        model = WaveLM_lit()
    elif cfg_file.startswith("Wave2Vec2"):
        model = Wav2Vec2_lit()
    elif cfg_file.startswith("LibriSeVoc"):
        model = LibriSeVoc_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("Ours/"):
        model = AudioModel_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("Wav2Clip/"):
        model = Wav2Clip_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("AudioClip/"):
        model = AudioClip_lit(cfg=cfg.MODEL)
    elif cfg_file.startswith("AASIST/"):
        model = AASIST_lit(cfg=cfg.MODEL)
    return model


# -

def make_callbacks(args):
    callbacks = [
        Color_progress_bar(),
        ACC_Callback(batch_key="label", output_key="pred"),
        AUC_Callback(batch_key="label", output_key="pred"),
        # ACC_Callback(batch_key="vocoder_label", output_key="vocoder_logit", num_classes=8, theme='vocoder'),
        # AUC_Callback(batch_key="vocoder_label", output_key="vocoder_logit", num_classes=8, theme='vocoder'),
        # ACC_Callback(batch_key="vocoder_label", output_key="content_voc_logit", num_classes=8, theme='vocoder_adv'),
        # AUC_Callback(batch_key="vocoder_label", output_key="content_voc_logit", num_classes=8, theme='vocoder_adv'),
        EER_Callback(batch_key="label", output_key="logit"),
        ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            monitor="val-auc",
            mode="max",
            save_last=True,
            filename="best-{epoch}-{val-auc:.2f}",
        ),
    ]
    if args.earlystop:
        callbacks.append(
            EarlyStopping(
                monitor="val-auc",
                min_delta=0.001,
                patience=args.earlystop if args.earlystop > 1 else 3,
                mode="max",
                stopping_threshold=0.999,
                verbose=True,
            ),
            # EarlyStopping(
            #     monitor="val-eer",
            #     min_delta=0.001,
            #     patience=args.earlystop if args.earlystop > 1 else 3,
            #     mode="min",
            #     stopping_threshold=0.001,
            #     verbose=True,
            # )
        )
    return callbacks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--ablation", type=str, default=None)
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("--batch_size", type=int, default=-1)
    parser.add_argument("--earlystop", type=int, default=3)
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    args = parser.parse_args()

    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=args.ablation
    )
    ds, dl = make_data(cfg.DATASET)

    model = make_model(args.cfg, cfg)
    callbacks = make_callbacks(args)

    
    trainer = pl.Trainer(
        max_epochs=cfg.MODEL.epochs,
        # max_epochs=1,
        accelerator="gpu",
        devices=args.gpu,
        logger=pl.loggers.CSVLogger(
            ROOT_DIR,
            name=args.cfg if args.ablation is None else args.cfg + "-" + args.ablation,
            version=args.version,
        ),
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        default_root_dir=ROOT_DIR,
        # profiler=pl.profilers.SimpleProfiler(dirpath='./', filename='test'),
        # limit_train_batches=500,
        # limit_val_batches=100,
        # limit_test_batches =100,
    )

    Console().print(
        "[on #00ff00][#ff3300]logger path : %s[/#ff3300][/on #00ff00]"
        % trainer.logger.log_dir
    )
    log_dir = trainer.logger.log_dir


    if not args.test:
        if args.resume:
            ckpt_path = get_ckpt_path(log_dir, theme="last")
            trainer.fit(model, dl.train, val_dataloaders=dl.val, ckpt_path=ckpt_path)
        else:
            trainer.fit(model, dl.train, val_dataloaders=dl.val)

    with open(os.path.join(log_dir, 'model.txt'), 'w') as f:
        s = summarize(model)
        f.write(str(s))
    
    backup_logger_file(log_dir)
    ckpt_path = get_ckpt_path(log_dir, theme="best")
    print(ckpt_path)
    model = model.load_from_checkpoint(ckpt_path, cfg=cfg.MODEL)
    for test_dl in to_list(dl.test):
        trainer.test(model, test_dl)
