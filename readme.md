# Dataset 

We use the WaveFake, LibriSeVoc, and DECRO dataset for training and test. You need to download them from their homepages.

After downloading, please change the paths for each dataset in  `config/datasets.py`:

```python
ROOT_PATHs = NameCfgNode(
    WaveFake="xxxx",
    LibriSeVoc="xxx",
    DECRO="xxxx",
)
```




# Training & Test


In train.py, you need to offer the ROOT_DIR for save model loggers and checkpoints.

```python
ROOT_DIR = "xxxx"
```

Besides, We use a global package in the `Packages`. You need to add the absolute path of the `Packages` into your `bashrc` or `zshrc`. 
```
export PYTHONPATH=$PYTHONPATH:path/of/Packages
```


Training cmds:

```bash
python train.py --gpu 1 --cfg 'Ours/DECRO_chinese';\
python train.py --gpu 1 --cfg 'Ours/DECRO_english';\
python train.py --gpu 1 --cfg 'Ours/LibriSeVoc_inner';\
python train.py --gpu 1 --cfg 'Ours/LibriSeVoc_cross_method';\
python train.py --gpu 1 --cfg 'Ours/LibriSeVoc_cross_dataset';\
python train.py --gpu 1 --cfg 'Ours/wavefake_inner';\
python train.py --gpu 1 --cfg 'Ours/wavefake_cross_lang';\
python train.py --gpu 1 --cfg 'Ours/wavefake_cross_method';\
python train.py --gpu 1 --cfg 'Ours/wavefake_cross_method2';\
```


Test cmds:
```bash
python train.py --gpu 0 --cfg 'Ours/DECRO_chinese' --test 1 --version 0;\
python train.py --gpu 0 --cfg 'Ours/DECRO_english' --test 1 --version 0;\
python train.py --gpu 0 --cfg 'Ours/LibriSeVoc_inner' --test 1 --version 0;\
python train.py --gpu 0 --cfg 'Ours/LibriSeVoc_cross_method' --test 1 --version 0;\
python train.py --gpu 0 --cfg 'Ours/LibriSeVoc_cross_dataset' --test 1 --version 0;\
python train.py --gpu 0 --cfg 'Ours/wavefake_inner' --test 1 --version 0;\
python train.py --gpu 0 --cfg 'Ours/wavefake_cross_lang' --test 1 --version 0;\
python train.py --gpu 0 --cfg 'Ours/wavefake_cross_method' --test 1 --version 0;\
python train.py --gpu 0 --cfg 'Ours/wavefake_cross_method2' --test 1 --version 0;\
```
