# [BMVC 2024] Distribution-Aware Calibration for Object Detection with Noisy Bounding Boxes

<div align=center>

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/pdf/2308.12017)
[![Code](https://img.shields.io/badge/Code-Github-purple?style=flat-square)](https://github.com/Correr-Zhou/DISCO)

</div>

**Abstract**: Large-scale well-annotated datasets are of great importance for training an effective object detector. However, obtaining accurate bounding box annotations is laborious and demanding. Unfortunately, the resultant noisy bounding boxes could cause corrupt supervision signals and thus diminish detection performance. Motivated by the observation that the real ground-truth is usually situated in the aggregation region of the proposals assigned to a noisy ground-truth, we propose **DIStribution-aware CalibratiOn** (**DISCO**) to model the spatial distribution of proposals for calibrating supervision signals.
In DISCO, spatial distribution modeling is performed to statistically extract the potential locations of objects. Based on the modeled distribution, three distribution-aware techniques, i.e., _distribution-aware proposal augmentation_ (_DA-Aug_), _distribution-aware box refinement_ (_DA-Ref_), and _distribution-aware confidence estimation_ (_DA-Est_), are developed to improve classification, localization, and interpretability, respectively. Extensive experiments demonstrate that DISCO can achieve SOTA performance in this task, especially at high noise levels.

## 🛠️ Installation
Please follow [OA-MIL](https://github.com/cxliu0/OA-MIL/) to prepare the environment and datasets.

## 🔬 Running
Run the following command to train and evaluate our DISCO on VOC dataset:
```
bash scripts/train_voc_disco.sh
```

Run the following command to train and evaluate our DISCO on COCO dataset:
```
bash scripts/train_coco_disco.sh
```

You can revise the noisy ratio and the corresponding hyperparameters of DISCO in these `.sh` scripts according to our paper.

## 📑 Citation
If you find that our work is helpful in your research, please consider citing our paper:
```latex
@article{zhou2023distribution,
  title={Distribution-Aware Calibration for Object Detection with Noisy Bounding Boxes},
  author={Zhou, Donghao and Li, Jialin and Li, Jinpeng and Huang, Jiancheng and Nie, Qiang and Liu, Yong and Gao, Bin-Bin and Wang, Qiong and Heng, Pheng-Ann and Chen, Guangyong},
  journal={arXiv preprint arXiv:2308.12017},
  year={2023}
}
```

## 🤝 Acknowledgement

Our code is built upon the repository of [mmdetection](https://github.com/open-mmlab/mmdetection) and [OA-MIL](https://github.com/cxliu0/OA-MIL/). Thank their authors for the excellent code.
