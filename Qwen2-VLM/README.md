---
license: Apache License 2.0
#用户自定义标签
tags:
- image caption
- MSCOCO

multi_modal:
  image-captioning:
    样本规模:
      - 10k-1m
    language:
      - en
    语言:
      - 英文
---
## 数据集描述
Image-caption task的数据集，包含train和valid

### 数据集简介
mscoco 2014的image caption数据集。

### 数据集支持的任务
支持image caption任务


## 数据集的格式和结构

### 数据格式
包含image_id, caption, image等信息。



### 数据集加载方式
```python
from modelscope.msdatasets import MsDataset
ds = MsDataset.load("coco_2014_caption", namespace="modelscope", split="train")
print(ds[0])
```

### 数据分片
数据已经预设了train/validation分片，是抽样得到的。



## 数据集生成的相关信息
原始数据参见：http://cocodataset.org/#home



## 数据集版权信息
数据集已经开源，暂未找到版权信息，如有违反相关条款，随时联系modelscope删除。

## 引用方式
```
@article{DBLP:journals/corr/LinMBHPRDZ14,
  author    = {Tsung{-}Yi Lin and
               Michael Maire and
               Serge J. Belongie and
               Lubomir D. Bourdev and
               Ross B. Girshick and
               James Hays and
               Pietro Perona and
               Deva Ramanan and
               Piotr Doll{'{a} }r and
               C. Lawrence Zitnick},
  title     = {Microsoft {COCO:} Common Objects in Context},
  journal   = {CoRR},
  volume    = {abs/1405.0312},
  year      = {2014},
  url       = {http://arxiv.org/abs/1405.0312},
  archivePrefix = {arXiv},
  eprint    = {1405.0312},
  timestamp = {Mon, 13 Aug 2018 16:48:13 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/LinMBHPRDZ14},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```


## 其他相关信息
数据源自ms coco 2014年的数据，可能存在bias，请合理使用。

### Clone with HTTP
* http://www.modelscope.cn/datasets/modelscope/coco_2014_caption.git
