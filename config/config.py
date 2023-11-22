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

# +
import os

from yacs.config import CfgNode as ConfigurationNode
# -

from .datasets import ALL_DATASETS, get_dataset_cfg

# # 默认配置

__C = ConfigurationNode()
__C.DATASET = ConfigurationNode()
__C.DATASET.name = "WaveFake"
__C.DATASET.task = "inner_eval"

# + editable=true slideshow={"slide_type": ""}
__C.MODEL = ConfigurationNode()
__C.MODEL.epochs = 200
__C.MODEL.optimizer = "AdamW"
__C.MODEL.weight_decay = 0.01
__C.MODEL.lr = 0.0001
__C.MODEL.lr_decay_factor = 0.5
__C.MODEL.lr_scheduler = "linear"
__C.MODEL.warmup_epochs = 10
__C.MODEL.label_smoothing = 0.1
__C.MODEL.method_classes = 7
__C.MODEL.pretrain = False
__C.MODEL.nograd = False
__C.MODEL.dims = [32, 64, 64,64, 128]
__C.MODEL.n_blocks = [1, 1, 1, 2, 1]
__C.MODEL.beta = [2, 0.5, 0.5]
__C.MODEL.one_stem = False


# -

def get_cfg_defaults(cfg_file=None, ablation=None):
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    res = __C.clone()

    if cfg_file is not None:
        res.merge_from_file(cfg_file)
        res.DATASET = get_dataset_cfg(
            name=res.DATASET.name, task=res.DATASET.task, __C=res.DATASET
        )

        aug_file_path = os.path.join(os.path.split(cfg_file)[0], "data_aug.yaml")
        if os.path.exists(aug_file_path):
            res.merge_from_file(aug_file_path)
            print("load aug yaml in ", aug_file_path)

        model_file_path = os.path.join(os.path.split(cfg_file)[0], "0-model.yaml")
        if os.path.exists(model_file_path):
            res.merge_from_file(model_file_path)
            print("load model yaml in ", model_file_path)

        if ablation is not None:
            ablation_file_path = os.path.join(os.path.split(cfg_file)[0], f"{ablation}.yaml")
            res.merge_from_file(ablation_file_path)
            print("load ablation yaml in ", ablation_file_path)
            

    for _ds in ALL_DATASETS:
        if _ds != res.DATASET.name:
            res.DATASET.pop(_ds)

    return res
