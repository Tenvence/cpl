<div align="center">
<h1> Controlling Class Layout for Deep Ordinal Classification via Constrained Proxies Learning
</center> <br> <center> </h1>

<p align="center">
Cong Wang, Zhiwei Jiang<sup>*</sup>, Yafeng Yin, Zifeng Cheng, Shiping Ge, Qing Gu
<br>
State Key Laboratory for Novel Software Technology, Nanjing University
<br>
<sup>*</sup> Corresponding author
<br><br>
AAAI 2023<br>
[<a href="https://doi.org/10.1016/j.cageo.2021.104912" target="_blank">paper</a>]
[<a href="https://doi.org/10.48550/arXiv.2303.00396" target="_blank">arXiv</a>]
[<a href="https://tenvence.github.io/files/cpl-poster.pdf" target="_blank">poster</a>] 
[<a href="https://tenvence.github.io/files/cpl-slides.pdf" target="_blank">slides</a>]
<br>
</div>

## Abstract

For deep ordinal classification, learning a well-structured feature space specific to ordinal classification is helpful to properly capture the ordinal nature among classes. Intuitively, when Euclidean distance metric is used, an ideal ordinal layout in feature space would be that the sample clusters are arranged in class order along a straight line in space. However, enforcing samples to conform to a specific layout in the feature space is a challenging problem. To address this problem, in this paper, we propose a novel Constrained Proxies Learning (CPL) method, which can learn a proxy for each ordinal class and then adjusts the global layout of classes by constraining these proxies. Specifically, we propose two kinds of strategies: hard layout constraint and soft layout constraint. The hard layout constraint is realized by directly controlling the generation of proxies to force them to be placed in a strict linear layout or semicircular layout (i.e., two instantiations of strict ordinal layout). The soft layout constraint is realized by constraining that the proxy layout should always produce unimodal proxy-to-proxies similarity distribution for each proxy (i.e., to be a relaxed ordinal layout). Experiments show that the proposed CPL method outperforms previous deep ordinal classification methods under the same setting of feature extractor.

## Reference
```
@inproceedings{wang2023controlling,
  title={Controlling Class Layout for Deep Ordinal Classification via Constrained Proxies Learning},
  author=Wang, Cong and Jiang, Zhiwei and Yin, Yafeng and Cheng, Zifeng and Ge, Shiping and Gu, Qing},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={2},
  pages={2483--2491},
  year={2023}
}
```
