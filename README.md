# Tri-modal Confluence with Temporal Dynamics for Scene Graph Generation in Operating Rooms (MICCAI 2024)
> [Paper](https://arxiv.org/pdf/2404.09231)
> 
> Authors:
> [Diandian Guo](https://scholar.google.com/citations?user=yXycwhIAAAAJ&hl=en), 
> [Manxi Lin](https://scholar.google.com/citations?user=RApnUsEAAAAJ&hl=da),
> [Jialun Pei](https://scholar.google.com/citations?user=1lPivLsAAAAJ&hl=en),
> [He Tang](https://scholar.google.com.hk/citations?user=70XLFUsAAAAJ&hl=zh-CN), 
> [Pheng-Ann Heng](https://scholar.google.com/citations?user=OFdytjoAAAAJ&hl=en).

## DEMO (click to play)
[![Watch the video](https://img.youtube.com/vi/amjobbBY89U/maxresdefault.jpg)](https://youtu.be/amjobbBY89U)

## Environment preparation

The code is tested on CUDA 11.1 and pytorch 1.9.0, change the versions below to your desired ones.

```shell
conda create -n tritempor python=3.7
conda install pytorch==1.9.0 torchvision cudatoolkit=11.1 -c pytorch -c nvidia -y
conda install cython scipy
pip install pycocotools
pip install opencv-python
```


### Pointnet++ preparation

Open3d is required for pointnet++. Please following the installation steps below to install open3d.

```shell
export CUDA=11.1
pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.9.0+${CUDA}.html
pip install torch-geometric
pip install open3d
```

Then refer to https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master for Pointnet++ installation. If you have any problems, feel free to contact us!



## Dataset preparation

### Download the datasets

- **Raw 4D-OR**: https://forms.gle/9cR3H5KcFUr5VKxr9

- **Processed 4D-OR**: [OneDrive](https://gocuhk-my.sharepoint.com/:f:/g/personal/jialunpei_cuhk_edu_hk/Es4MUdHVUE1LpOJn2vHQFKMBNGfr2O0LT0xLG8HLMFVWEg?e=ovg5RL)

### Datasets for training

Download the Processed 4D-OR provided above. The data folder should be like this:

```shell
TriTemp-OR/data/: 
              /images/: unzip 4d_or_images_multiview_reltrformat.zip
              /points/: unzip points.zip
              /infer/: unzip infer.zip
              /train.json: from reltr_annotations_8.3.zip
              /val.json: from reltr_annotations_8.3.zip
              /test.json: from reltr_annotations_8.3.zip
              /rel.json: from reltr_annotations_8.3.zip
```

## Pre-trained models

Pretrained DETR weights: [OneDrive](https://gocuhk-my.sharepoint.com/:u:/g/personal/jialunpei_cuhk_edu_hk/EcQvVowPUVBItg8tIM1L7SMBXGQu4xQXTSrZNOcNSDHIwg?e=pFkhQx).


## Usage

### Train

```shell
python -m torch.distributed.launch --nproc_per_node={num_gpus} --use_env main.py --validate \
--num_hoi_queries 100 --batch_size 2 --lr 5e-5 --hoi_aux_loss --dataset_file or \
--detr_weights {pretrained DETR path}  --output_dir {output_path} --group_name {output_group_name} \
--HOIDet --run_name {output_run_name} --epochs 100 --ann_path /data/4dor/ --img_folder /data/4dor/images \
--num_queries 20 --use_tricks_val --use_relation_tgt_mask --add_none --train_detr --use_pointsfusion \
--use_multiview_fusion --use_multiviewfusion_last_view2
```

### Inference

```shell
python -m torch.distributed.launch --nproc_per_node={num_gpus} --use_env main.py --validate \
--num_hoi_queries 100 --batch_size 2 --lr 5e-5 --hoi_aux_loss --dataset_file or \
--detr_weights {pretrained DETR path}  --output_dir {output_path} --group_name {output_group_name} \
--HOIDet --run_name {output_run_name} --epochs 100 --ann_path /data/4dor/ --img_folder /data/4dor/images \
--num_queries 20 --use_tricks_val --use_relation_tgt_mask --add_none --use_pointsfusion \
--use_multiview_fusion --use_multiviewfusion_last_view2 --resume {MODEL_WEIGHTS} --infer
```

Please replace `{MODEL_WEIGHTS}` to the pre-trained weights

### Visualization




## Acknowledgement

This work is based on:
- [STIP](https://github.com/zyong812/STIP)
- [RelTR](https://github.com/yrcong/RelTR)
- [DETR](https://github.com/facebookresearch/detr)


Thanks for their great work!

## Citation

If this helps you, please cite this work:

```bibtex
@inproceedings{guo2024tri,
  title={Tri-modal Confluence with Temporal Dynamics for Scene Graph Generation in Operating Rooms},
  author={Guo, Diandian and Lin, Manxi and Pei, Jialun and Tang, He and Jin, Yueming and Heng, Pheng-Ann},
  booktitle={MICCAI},
  year={2024},
  organization={Springer}
}
```
