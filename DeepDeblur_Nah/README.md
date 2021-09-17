If you find our work useful in your research or publication, please cite our work:
```
@InProceedings{Nah_2017_CVPR,
  author = {Nah, Seungjun and Kim, Tae Hyun and Lee, Kyoung Mu},
  title = {Deep Multi-Scale Convolutional Neural Network for Dynamic Scene Deblurring},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {July},
  year = {2017}
}
```
## Usage examples

* Preparing dataset

Before running the code, put the datasets on a desired directory. By default, the data root is set as '~/Research/dataset'  
See: [src/option.py](src/option.py)
```python
group_data.add_argument('--data_root', type=str, default='~/Research/dataset', help='dataset root location')
```
Put your dataset under ```args.data_root```.

The dataset location should be like:
```bash
# GOPRO_Large dataset
~/Research/dataset/GOPRO_Large/train/GOPR0372_07_00/blur_gamma/....
# REDS dataset
~/Research/dataset/REDS/train/train_blur/000/...
```

* Example commands

```bash
# single GPU training
python main.py --n_GPUs 1 --batch_size 8 # save the results in default experiment/YYYY-MM-DD_hh-mm-ss
python main.py --n_GPUs 1 --batch_size 8 --save_dir GOPRO_L1  # save the results in experiment/GOPRO_L1

# adversarial training
python main.py --n_GPUs 1 --batch_size 8 --loss 1*L1+1*ADV
python main.py --n_GPUs 1 --batch_size 8 --loss 1*L1+3*ADV
python main.py --n_GPUs 1 --batch_size 8 --loss 1*L1+0.1*ADV

# train with GOPRO_Large dataset
python main.py --n_GPUs 1 --batch_size 8 --dataset GOPRO_Large
# train with REDS dataset (always set --do_test false)
python main.py --n_GPUs 1 --batch_size 8 --dataset REDS --do_test false --milestones 100 150 180 --end_epoch 200

# save part of the evaluation results (default)
python main.py --n_GPUs 1 --batch_size 8 --dataset GOPRO_Large --save_results part
# save no evaluation results (faster at test time)
python main.py --n_GPUs 1 --batch_size 8 --dataset GOPRO_Large --save_results none
# save all of the evaluation results
python main.py --n_GPUs 1 --batch_size 8 --dataset GOPRO_Large --save_results all
```

```bash
# multi-GPU training (DataParallel)
python main.py --n_GPUs 2 --batch_size 16
```

```bash
# multi-GPU training (DistributedDataParallel), recommended for the best speed
# single command version (do not set ranks)
python launch.py --n_GPUs 2 main.py --batch_size 16

# multi-command version (type in independent shells with the corresponding ranks, useful for debugging)
python main.py --batch_size 16 --distributed true --n_GPUs 2 --rank 0 # shell 0
python main.py --batch_size 16 --distributed true --n_GPUs 2 --rank 1 # shell 1
```

```bash
# single precision inference (default)
python launch.py --n_GPUs 2 main.py --batch_size 16 --precision single

# half precision inference (faster and requires less memory)
python launch.py --n_GPUs 2 main.py --batch_size 16 --precision half

# half precision inference with AMP
python launch.py --n_GPUs 2 main.py --batch_size 16 --amp true
```

```bash
# optional mixed-precision training
# mixed precision training may result in different accuracy
python main.py --n_GPUs 1 --batch_size 16 --amp true
python main.py --n_GPUs 2 --batch_size 16 --amp true
python launch.py --n_GPUs 2 main.py --batch_size 16 --amp true
```

```bash
# Advanced usage examples 
# using launch.py is recommended for the best speed and convenience
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --end_epoch 1000 --save_results none
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --end_epoch 1000 --save_results part
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --end_epoch 1000 --save_results all
python launch.py --n_GPUs 4 main.py --dataset GOPRO_Large --milestones 500 750 900 --end_epoch 1000 --save_results all --amp true

python launch.py --n_GPUs 4 main.py --dataset REDS --milestones 100 150 180 --end_epoch 200 --save_results all --do_test false
python launch.py --n_GPUs 4 main.py --dataset REDS --milestones 100 150 180 --end_epoch 200 --save_results all --do_test false --do_validate false
```

```bash
# Commands used to generate the below results
python launch.py --n_GPUs 2 main.py --dataset GOPRO_Large --milestones 500 750 900 --end_epoch 1000
python launch.py --n_GPUs 4 main.py --dataset REDS --milestones 100 150 180 --end_epoch 200 --do_test false
```

For more advanced usage, please take a look at src/option.py
