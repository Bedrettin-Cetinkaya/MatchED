# MatchED: MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision ( CVPR 2026)

## 1. Install Environment, Datasets
### 1.1 Environment
Please follow this repository to set the environment: https://github.com/hellozhuo/pidinet

### 1.2 Datasets
#### NYUD:

Download NYUD data from [EDTER](https://github.com/MengyangPu/EDTER) Repository:

-Download [Train Data](https://drive.google.com/drive/folders/1lTfTIS-vlTtId-LGhEO2ZZjonA3SmLGJ)

-Download [Test Data](https://drive.google.com/drive/folders/1TQpKzCV4Ujkfs4V_vMasKAcvg3p4ByCN)

Put these files into data/NYUD/.

Use our training and testing txt files.

#### BSDS:

-Download BSDS data from [here](https://drive.google.com/drive/folders/16W1yK8LpbJNin5C8_HLTt_w5pz25kLOs?usp=sharing)
-Put these files into data/BSDS/.

-Download BSDS gt .mat files from [here](https://drive.google.com/file/d/1DFHDpVMc32qBAIRn6o4QCqg5RupV2D3x/view?usp=sharing)
-Put these files into data/BSDS/train/gt_mat

Extract these files via tar zxvf filename.tar.gz

## 2. Train Model
### 2.1 NYU:
-Change max_toleration and weight (in Lines 189 and 190 )  [https://github.com/Bedrettin-Cetinkaya/MatchED/blob/main/utils.py#L182)
        
-Run the following command to start training.

```shell
python main.py --model pidinet --config carv4 --sa --dil --iter-size 24 -j 4 --gpu 0 --epochs 15 --lr 0.005 --lr-type multistep --lr-steps 8-12 --wd 1e-4 --savedir your_save_dir --datadir your_data_dir/NYUD --dataset NYUD-image

```

### 2.2 BSDS:

#### For Only Ranking:
-Change max_toleration and weight (in Lines 189 and 190 )  [https://github.com/Bedrettin-Cetinkaya/MatchED/blob/main/utils.py#L182)
        
-Run the following command to start training.

```shell
python main.py --model pidinet --config carv4 --sa --dil --iter-size 24 -j 4 --gpu 0 --epochs 15 --lr 0.005 --lr-type multistep --lr-steps 8-12 --wd 1e-4 --savedir your_save_dir --datadir your_data_dir/BSDS --dataset BSDS

```

## 3. Inference
-Run the following command to start inference. 
python main.py --model pidinet --config carv4 --sa --dil --iter-size 24 -j 4 --gpu 0 --epochs 15 --lr 0.005 --lr-type multistep --lr-steps 8-12 --wd 1e-4 --savedir your_save_dir --datadir your_data_dir/NYUD --dataset NYUD-image --evaluate your_save_dir/save_models/checkpoint_013.pth


## 4. Pre-trained Models:
-https://drive.google.com/file/d/1JV5P2O8j8pTH6F70QjQRg7mw08sGvwC5/view?usp=sharing

## 4. Acknowledgements
Thanks to the previous open-sourced repo:

[PiDiNet](https://github.com/MengyangPu/EDTER)


## 5. Reference
```bibtex
@InProceedings{cetinkaya2026matched,
  title={MatchED: Crisp Edge Detection Using End-to-End, Matching-based Supervision}, 
  author={Bedrettin Cetinkaya and Sinan Kalkan and Emre Akbas},
  year={2026},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  url={https://cvpr26-matched.github.io/}
}
```

