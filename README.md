# [BMVC 2024] Distribution-Aware Calibration for Object Detection with Noisy Bounding Boxes

<div align=center>

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b?style=flat-square)](https://arxiv.org/pdf/2308.12017)
[![Code](https://img.shields.io/badge/Code-Github-purple?style=flat-square)](https://github.com/Correr-Zhou/DISCO)

</div>

**Abstract**: Large-scale well-annotated datasets are of great importance for training an effective object detector. However, obtaining accurate bounding box annotations is laborious and demanding. Unfortunately, the resultant noisy bounding boxes could cause corrupt supervision signals and thus diminish detection performance. Motivated by the observation that the real ground-truth is usually situated in the aggregation region of the proposals assigned to a noisy ground-truth, we propose **DIStribution-aware CalibratiOn** (**DISCO**) to model the spatial distribution of proposals for calibrating supervision signals.
In DISCO, spatial distribution modeling is performed to statistically extract the potential locations of objects. Based on the modeled distribution, three distribution-aware techniques, \textit{i.e.}, _distribution-aware proposal augmentation_ (_DA-Aug_), _distribution-aware box refinement_ (_DA-Ref_), and _distribution-aware confidence estimation_ (_DA-Est_), are developed to improve classification, localization, and interpretability, respectively. Extensive experiments demonstrate that DISCO can achieve SOTA performance in this task, especially at high noise levels.

## 🔥 Updates
- 2024.07: We are delighted to announce that this paper was accepted by BMVC 2024! Code is coming soon!
