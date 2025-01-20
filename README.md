## News
This paper is online at https://doi.org/10.1007/s00371-024-03757-w in The Visual Computer.               

## SSP AD Dataset

if you would like to access the dataset, please feel free to contact us via email at crl1567@163.com.     
---

## Quick Guide

First, clone this repository and set the `PYTHONPATH` environment variable with `env PYTHONPATH=src python bin/run_patchcore.py`.
To train PatchCore on MVTec AD (as described below), run

```
datapath=/media/ubuntu/lihui/detect_defect/datasets/mvtec
#aug_datapath=/media/ubuntu/data/crl1/mvtec_sci_easy
aug_datapath=/media/ubuntu/data/crl1/4_mvtec/mvtec_sci_easy
datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
#datasets=( 'carpet' 'grid'  'leather' 'tile' 'wood')
#datasets=( 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')

dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '"${dataset}"; done))
python src/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0  --log_project MVTecAD_Results results \
patch_core -b wideresnet50  -le layer2 -le layer3  --faiss_on_gpu --pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath  aug_dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $aug_datapath
```

## In-Depth Description

### Requirements

Our results were computed using Python 3.8, with packages and respective version noted in
`requirements.txt`. In general, the majority of experiments should not exceed 11GB of GPU memory;
however using significantly large input images will incur higher memory cost.

### Setting up MVTec AD

To set up the main MVTec AD benchmark, download it from here: <https://www.mvtec.com/company/research/datasets/mvtec-ad>.
Place it in some location `datapath`. Make sure that it follows the following data tree:

```shell
mvtec
|-- bottle
|-----|----- ground_truth
|-----|----- test
|-----|--------|------ good
|-----|--------|------ broken_large
|-----|--------|------ ...
|-----|----- train
|-----|--------|------ good
|-- cable
|-- ...
```

containing in total 15 subdatasets: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`,
`leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`.

### Evaluating a pretrained PatchCore model

To evaluate a/our pretrained PatchCore model(s), run

```shell
python bin/load_and_evaluate_patchcore.py --gpu <gpu_id> --seed <seed> $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
```

assuming your pretrained model locations to be contained in `model_flags`; one for each subdataset
in `dataset_flags`. Results will then be stored in `savefolder`. Example model & dataset flags:

```shell
model_flags=('-p', 'path_to_mvtec_bottle_patchcore_model', '-p', 'path_to_mvtec_cable_patchcore_model', ...)
dataset_flags=('-d', 'bottle', '-d', 'cable', ...)
```


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.


## Citation
If you find these projects useful, please consider citing:

```shell
@article{cao2024dual,
  title={Dual-flow feature enhancement network for robust anomaly detection in stainless steel pipe welding},
  author={Cao, Runlong and Zhang, Jianqi and Shen, Yun and Zhou, Huanhuan and Zhou, Peiying and Shen, Guowei and Xia, Zhengwen and Zang, Ying and Liu, Qingshan and Hu, Wenjun},
  journal={The Visual Computer},
  pages={1--17},
  year={2024},
  publisher={Springer}
}
```
