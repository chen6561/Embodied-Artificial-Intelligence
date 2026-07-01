# 导入深度学习基础库torch，用于张量运算、模型设备调度
import torch
# 导入json库，用于读取、保存本地json格式多模态对话数据集
import json
# 导入swanlab可视化工具，用于训练指标、预测样例在线日志记录
import swanlab
# 导入Hugging Face Dataset，用于构建、映射、加载结构化训练数据集
from datasets import Dataset
# modelscope工具：自动下载开源模型权重；AutoTokenizer通用分词器加载工具
from modelscope import snapshot_download, AutoTokenizer
# transformers框架swanlab集成回调，自动上传loss、lr、epoch等训练指标
from swanlab.integration.transformers import SwanLabCallback
# Qwen2-VL官方工具函数，统一处理多模态输入图像信息、切分图像块
from qwen_vl_utils import process_vision_info
# PEFT库LoRA微调相关：LoRA配置、任务类型、绑定LoRA到基座模型、加载微调后LoRA权重
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
# transformers训练全套工具包
from transformers import (
    TrainingArguments,          # 训练超参配置类
    Trainer,                    # Hugging Face标准训练器
    DataCollatorForSeq2Seq,     # 序列生成任务专用数据填充器
    Qwen2VLForConditionalGeneration, # Qwen2-VL多模态图文生成模型
    AutoProcessor,              # Qwen2-VL专用多模态处理器（图像+文本联合编码）
)

def process_func(example):
    """
    数据预处理函数：将原始json对话样本，转换成模型可直接输入的张量格式
    输入：单条原始数据集样本dict，包含conversations对话字段
    输出：结构化张量字典：input_ids/attention_mask/labels/图像像素/图像分块尺寸
    核心逻辑：构造用户图文prompt、编码文本+图像、构建带-100掩码的损失标签（仅计算回答损失）
    """
    # 输入输出总长度上限，超长样本截断，防止显存溢出
    MAX_LENGTH = 8192
    # 取出单条样本完整对话列表，0号为用户图文输入，1号为模型回答输出
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]

    # 从用户文本中提取图片文件路径，文本格式自带<|vision_start|><|vision_end|>图像占位符
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

    # 组装Qwen2-VL标准多模态消息格式：1张图 + 固定提示文本
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",       # 多模态类型：图片
                    "image": file_path,     # 本地图片文件路径
                    "resized_height": 280, # 图像统一缩放高度
                    "resized_width": 280,  # 图像统一缩放宽度
                },
                {"type": "text", "text": "COCO Yes:"}, # 任务提示词，适配COCO图像描述任务
            ],
        }
    ]

    # 使用processor内置对话模板拼接prompt文本，不进行分词，返回完整字符串
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True # 自动添加模型生成起始标记
    )

    # 解析消息内的图像，返回图像加载对象、图像元信息（此处仅取图像输入）
    image_inputs, _ = process_vision_info(messages)
    # 多模态联合编码：文本+图像统一转为模型输入张量
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt" # 返回PyTorch张量格式
    )

    # 单独对模型回答文本分词，不添加特殊首尾标记，作为标签部分
    response = tokenizer(
        output_content,
        add_special_tokens=False,
        return_tensors="pt"
    )

    # 分离prompt（用户输入）的token ids与注意力掩码，去除batch维度
    prompt_input_ids = inputs["input_ids"][0]
    prompt_attention_mask = inputs["attention_mask"][0]

    # 分离模型回答的token ids与注意力掩码，去除batch维度
    response_input_ids = response["input_ids"][0]
    response_attention_mask = response["attention_mask"][0]

    # 定义padding填充token id与对应掩码标记
    pad_id = torch.tensor([tokenizer.pad_token_id], dtype=prompt_input_ids.dtype)
    pad_mask = torch.tensor([1], dtype=prompt_attention_mask.dtype)

    # ====================== 拼接完整输入与构造损失标签 ======================
    # 输入序列 = 用户prompt tokens + 模型回答tokens + 末尾填充token
    input_ids = torch.cat(
        [prompt_input_ids, response_input_ids, pad_id],
        dim=0
    )
    # 注意力掩码：全部有效token置1，填充位也标记为有效
    attention_mask = torch.cat(
        [prompt_attention_mask, response_attention_mask, pad_mask],
        dim=0
    )
    # label构造关键：用户prompt部分全部填充-100（交叉熵会自动忽略，不计算损失）
    # 仅模型回答部分作为有效标签参与loss计算，末尾补填充token
    labels = torch.cat(
        [
            torch.full((prompt_input_ids.size(0),), -100, dtype=prompt_input_ids.dtype),
            response_input_ids,
            pad_id
        ],
        dim=0
    )

    # 超长样本截断，统一限制最大序列长度
    if input_ids.size(0) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    # 返回模型训练所需全部字段：文本token、掩码、损失标签、图像像素、图像分块尺寸
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs["pixel_values"].squeeze(0), # 去除batch维度的图像像素张量
        "image_grid_thw": inputs["image_grid_thw"].squeeze(0), # 图像分块尺寸参数
    }

def predict(messages, model):
    """
    推理预测函数：训练完成后加载LoRA权重，输入图文对话，返回模型生成的图像描述文本
    :param messages: Qwen2-VL标准多模态消息列表（图片+文本prompt）
    :param model: 加载LoRA微调权重后的多模态模型
    :return: 模型生成的图像描述字符串
    """
    # 套用对话模板拼接输入文本
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 解析加载图像
    image_inputs, _ = process_vision_info(messages)
    # 多模态编码并迁移至GPU
    inputs = processor(text=[text], images=image_inputs,
                      padding=True, return_tensors="pt").to("cuda")
    
    # 执行生成，限制最大新生成token数量128
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    # 截断输入prompt部分，只保留模型新生成的token
    generated_ids_trimmed = [
        out_ids[len(in_ids):] 
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    # 解码token为文本，跳过特殊符号、不清理多余空格，返回第一条生成结果
    return processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )[0]

# ====================== 1. 基座模型、分词器、多模态处理器加载 ======================
# modelscope自动下载Qwen2-VL-2B-Instruct基座模型，缓存到本地./目录
model_dir = snapshot_download("Qwen/Qwen2-VL-2B-Instruct", 
                             cache_dir="./", revision="master")
# 加载专用分词器，关闭快速分词，开启自定义代码信任
tokenizer = AutoTokenizer.from_pretrained(
    "./Qwen/Qwen2-VL-2B-Instruct/", use_fast=False, trust_remote_code=True
)
# 加载Qwen2-VL多模态处理器（统一处理图像+文本编码）
processor = AutoProcessor.from_pretrained("./Qwen/Qwen2-VL-2B-Instruct")
# 加载多模态图文生成基座模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./Qwen/Qwen2-VL-2B-Instruct/", 
    device_map="auto",        # 自动分配模型层到GPU/CPU
    torch_dtype=torch.bfloat16, # 使用bf16混合精度训练，节省显存
    trust_remote_code=True    # 信任模型自定义代码
)
# 开启输入梯度计算，LoRA微调必备配置
model.enable_input_require_grads()

# ====================== 2. 数据集划分：训练集/测试集拆分并本地保存 ======================
# 读取原始多模态对话数据集
with open("data_vl.json", 'r') as f:
    data = json.load(f)
    train_data = data[:-4]  # 前496条作为训练集
    test_data = data[-4:]   # 最后4条作为离线验证测试集，用于可视化效果

# 拆分后的训练集写入本地json文件
with open("data_vl_train.json", "w") as f:
    json.dump(train_data, f)
# 拆分后的测试集写入本地json文件
with open("data_vl_test.json", "w") as f:
    json.dump(test_data, f)

# 从本地json构建Hugging Face Dataset对象
train_ds = Dataset.from_json("data_vl_train.json")
# 批量映射预处理函数，把原始对话转为模型张量输入
train_dataset = train_ds.map(process_func)

# ====================== 3. LoRA微调超参配置 ======================
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, # 任务类型：自回归语言生成
    # 需要注入LoRA的Transformer全量权重层，覆盖注意力、MLP模块
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, # 训练模式关闭推理模式
    r=64,                 # LoRA低秩矩阵维度
    lora_alpha=16,        # LoRA缩放系数
    lora_dropout=0.05,    # LoRA层dropout概率
    bias="none",          # 不训练bias偏置参数
)
# 将LoRA适配器绑定到基座模型，冻结主干仅训练LoRA权重
peft_model = get_peft_model(model, config)

# ====================== 4. 训练全局超参配置 ======================
args = TrainingArguments(
    output_dir="./output/Qwen2-VL-2B", # 模型checkpoint保存根目录
    per_device_train_batch_size=4,     # 单卡batch size
    gradient_accumulation_steps=4,     # 梯度累积步数，等效batch=4*4=16
    logging_steps=10,                  # 每10步打印一次训练日志
    logging_first_step=5,              # 前5步即输出第一条日志
    num_train_epochs=2,                # 完整训练轮次
    save_steps=100,                    # 每100步保存一次模型断点
    learning_rate=1e-4,                # LoRA训练学习率
    save_on_each_node=True,            # 多机分布式场景每节点保存断点
    gradient_checkpointing=True,       # 梯度检查点，大幅降低显存占用
    report_to="none",                  # 关闭transformers默认日志工具，改用swanlab
)

# ====================== 5. SwanLab可视化日志回调配置 ======================
swanlab_callback = SwanLabCallback(
    project="Qwen2-VL-finetune",        # swanlab项目分组名
    experiment_name="qwen2-vl-coco2014",# 当前实验名称
    config={ # 记录本次实验全部关键超参，方便复现对比
        "model": "Qwen2-VL-2B-Instruct",
        "dataset": "coco_2014_caption",
        "train_data_number": len(train_data),
        "lora_rank": 64,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
    },
)

# ====================== 6. 初始化Trainer并启动LoRA微调训练 ======================
trainer = Trainer(
    model=peft_model,                          # 绑定LoRA的微调模型
    args=args,                                 # 训练超参
    train_dataset=train_dataset,              # 预处理完成的训练数据集
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True), # 序列生成专用填充器
    callbacks=[swanlab_callback],              # 挂载swanlab日志回调
)
# 启动完整训练流程
trainer.train()

# ====================== 7. 加载训练断点LoRA权重，离线推理测试 ======================
# 推理模式LoRA配置，仅加载权重不参与梯度更新
val_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    inference_mode=True, # 开启推理模式，冻结梯度
    r=64, lora_alpha=16, lora_dropout=0.05, bias="none",
)
# 基于原始基座模型，加载训练产出的LoRA断点权重
val_peft_model = PeftModel.from_pretrained(
    model, model_id="./output/Qwen2-VL-2B/checkpoint-62", config=val_config
)

# 存储swanlab可视化图片对象，用于批量上传预测结果
test_image_list = []
# 读取测试集4条样本
with open("data_vl_test.json", "r") as f:
    test_dataset = json.load(f)

# 遍历每条测试样本执行推理
for item in test_dataset:
    # 从对话文本中提取原图路径
    origin_image_path = item["conversations"][0]["value"].split(
        "<|vision_start|>")[1].split("<|vision_end|>")[0]
    
    # 构造推理用多模态prompt
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": origin_image_path},
            {"type": "text", "text": "COCO Yes:"}
        ]
    }]
    
    # 调用推理函数生成图像描述
    response = predict(messages, val_peft_model)
    # 控制台打印预测结果
    print(f"生成结果：{response}")
    # 封装swanlab图像日志对象，图片+生成文本caption
    test_image_list.append(swanlab.Image(origin_image_path, caption=response))

# 将全部测试图片+预测文本批量上传至swanlab可视化面板
swanlab.log({"Prediction": test_image_list})
# 结束swanlab实验日志记录
swanlab.finish()