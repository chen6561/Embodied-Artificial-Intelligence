# coding=utf-8
"""COCO2014 图像描述数据集 Hugging Face datasets 自定义加载器"""
# 导入csv模块，用于读取csv格式的COCO标注文件
import csv
# 导入huggingface datasets库，用于自定义数据集构建、管理、加载
import datasets

# ====================== 论文引用信息 ======================
# 数据集对应的官方论文BibTeX引用格式，用于学术实验成果标注
_CITATION = """\
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
"""

# ====================== 数据集描述文本 ======================
# 数据集官方简介，会在dataset.info中对外展示
_DESCRIPTION = """\
COCO 是大规模通用视觉数据集，支持目标检测、实例分割、全景分割、图像描述多类视觉任务。
COCO2014Caption 为该数据集的图像字幕子任务子集，每张图片配套多条自然语言描述文本。
数据集包含328k张实拍自然场景图像，覆盖91类日常物体，配套250万+物体实例标注。
"""

# ====================== 数据集官方主页地址 ======================
_HOMEPAGE = "http://cocodataset.org/#home"

# ====================== 数据集开源协议 ======================
# COCO数据集采用 CC-BY 4.0 协议，允许商用、二次分发，需标注原作者来源
_LICENSE = "cc-by-4.0"

# ====================== 数据集远程下载地址 ======================
# 字典存储训练集/验证集的csv压缩包OSS下载链接
_URLS = {
    "train": "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/coco_2014_caption/train2014.csv.zip",
    "valid": "https://modelscope.oss-cn-beijing.aliyuncs.com/open_data/coco_2014_caption/val2014.csv.zip"
}


class COCO2014Caption(datasets.GeneratorBasedBuilder):
    """
    COCO2014 图像字幕数据集自定义加载类
    继承datasets.GeneratorBasedBuilder，通过生成器逐行读取csv构建样本
    任务目标：输入图像，输出对应自然语言描述caption，适用于图像字幕训练
    """

    # 数据集配置列表，支持多版本/多子集，当前仅定义coco_2014_caption基础版本
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="coco_2014_caption",  # 数据集唯一标识名称
            version=datasets.Version("1.0.0"),  # 数据集版本号，更新数据时迭代版本
            description=_DESCRIPTION,  # 绑定数据集描述文本
        )
    ]

    # 定义单条样本包含的全部字段与数据类型
    features = {
                "uniq_id": datasets.Value("string"),  # 样本全局唯一ID，字符串格式
                "image_id": datasets.Value("string"), # 对应COCO原图的图像编号
                "caption": datasets.Value("string"),  # 图像对应的自然语言描述文本
                "image": datasets.Image(),            # 原图图像字段，自动解析图片二进制
            }

    def _info(self):
        """
        数据集元信息构建函数
        返回DatasetInfo对象，存储数据集全部基础信息，供外部调用查看
        """
        return datasets.DatasetInfo(
            description=_DESCRIPTION,    # 数据集介绍
            features=datasets.Features(self.features), # 样本字段定义
            supervised_keys=None,        # 无标准监督输入输出对，图像字幕任务自行匹配
            homepage=_HOMEPAGE,          # 官方网站链接
            license=_LICENSE,            # 开源协议
            citation=_CITATION,          # 论文引用信息
        )

    def _split_generators(self, dl_manager):
        """
        数据集分片下载与解压函数
        dl_manager：内置下载管理器，自动完成远程文件下载、本地解压
        返回训练集、验证集两个分片的生成器配置
        """
        # 根据_URLS下载并自动解压zip包，返回解压后的文件路径字典
        data_files = dl_manager.download_and_extract(_URLS)
        return [
            # 训练集分片生成器配置
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "files": dl_manager.iter_files([data_files["train"]]), # 迭代读取训练集所有解压文件
                },
            ),
            # 验证集分片生成器配置
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "files": dl_manager.iter_files([data_files["valid"]]), # 迭代读取验证集所有解压文件
                },
            ),
        ]

    def _generate_examples(self, files):
        """
        样本生成器核心函数，逐行读取csv文件生成单条数据样本
        :param files: 迭代器，遍历当前分片下所有csv文件
        :yield: 索引值 + 单条样本字典，符合datasets库标准输出格式
        """
        # 全局样本自增索引，作为每条样本的唯一主键
        idx = 0
        # 遍历分片下所有csv文件
        for file_name in files:
            # utf8编码打开csv标注文件
            with open(file_name, encoding="utf8") as f:
                # 读取csv表头，自动映射列名与行数据
                reader = csv.DictReader(f)
                # 逐行遍历csv标注
                for row in reader:
                    # 按预定义features字段筛选、组装样本字典
                    example = {feat: row[feat] for feat in self.features.keys()}
                    # 生成一条样本，idx为样本编号，example为完整数据
                    yield idx, example
                    # 索引自增
                    idx += 1