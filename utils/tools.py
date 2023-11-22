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

import os
import datetime
import shutil


def backup_logger_file(logger_version_path):
    
    metric_file = os.path.join(logger_version_path, 'metrics.csv')
    m_time = os.path.getmtime(metric_file)
    m_time = datetime.datetime.fromtimestamp(m_time)
    m_time = m_time.strftime('%Y-%m-%d-%H:%M:%S')

    if os.path.exists(metric_file):
        backup_file = metric_file.replace('.csv', f'-{m_time}.csv')
        if not os.path.exists(backup_file):
            shutil.copy2(metric_file, backup_file)


def get_ckpt_path(logger_dir, theme='best'):
    checkpoint = os.path.join(logger_dir, "checkpoints")
    for path in os.listdir(checkpoint):
        if theme in path:
            ckpt_path = os.path.join(checkpoint, path)
            return ckpt_path
    raise FileNotFoundError(f'There are no {theme} ckpt in {logger_dir}')    

# + tags=["active-ipynb", "style-student"]
# path = '/usr/local/ay_data/1-model_save/3-CS/CSNet+/coco/1/version_0'
# backup_logger_file(path)
