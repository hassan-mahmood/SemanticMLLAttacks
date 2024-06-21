# Semantic-Aware Multi-Label Adversarial Attacks

## Overview
This repository contains the implementation of [Semantic-Aware Multi-Label Adversarial Attacks](https://openaccess.thecvf.com/content/CVPR2024/papers/Mahmood_Semantic-Aware_Multi-Label_Adversarial_Attacks_CVPR_2024_paper.pdf)

> In this work, we propose an optimization that searches for an attack perturbation that modifies the predictions of labels in the consistent target set while ensuring that other labels will not get affected. This leads
to an efficient algorithm that projects the gradient of the consistent target set loss onto the orthogonal direction of the gradient of the loss on other labels.

## Data Preparation
Following are the details to replicate the results on OpenImages. This is applicable to any multi-label dataset.

1) Download the test images and store them in a folder 'test_images_folders'. 
2) Store test labels in binary format (one-hot encoding) and image IDs corresponding to the test labels in an hdf format file i.e., each row of the hdf file: [0, 1, 0, 0, ..., 'temp/img0212.jpg'].

3) Store the classnames in a pickle file as a list. Download the semantic relationship hierarchy (tree) file.

4) Generate a list of different target labels to be attacked. For example, [[1,2,3],[5,6,7],[2,3]] are 3 sets of target labels and the running the code will iterate through each set, select the associated images and generate the attack.

5) Update fields/values in configs/oi.ini. Update 'weights_dir'(to store any weights), 'log_folder'(to store the experiment log), and 'pred_file'(add path of the labels file). Also update the 'test_labels_files', 'test_images_folders', 'classnames_path', 'tree_path', and 'target_labels_path'.


## Model Preparation
Any model trained on the target multi-label data can be used. For OpenImages, we use ASL-based TResNet-L model.

1) Download the model weights from the official repo of ASL: [model weights](https://github.com/Alibaba-MIIL/ASL/blob/main/MODEL_ZOO.md) in current code folder.

## Generating the Attack

1) To generate the attack, run the following command:
```python aslscripts/consistent_mll_asl_oi_TurnOFF.py --configfile configs/oi.ini asl```

## Citation
```
 @InProceedings{Mahmood_2024_CVPR,
    author    = {Mahmood, Hassan and Elhamifar, Ehsan},
    title     = {Semantic-Aware Multi-Label Adversarial Attacks},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision       and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {24251-24262}
}
```

