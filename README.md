# 1. Use BasicSR Framework to train classifier
- data：构建自己的 dataloader
    - dataloader 是公用的，实质上指的是创建自己的Dataset Class
    - 基于__init__,\_\_getitem\_\_,\_\_len\_\_这些基本函数
    - 核心逻辑放在\_\_getitem\_\_中，它指定了每次迭代时需要返回什么
    - basicsr 提供了丰富的数据类和数据处理函数：
        - basicsr/data/degradations
        - basicsr/data/tranforms 
        - basicsr/data/data_util
- arch：定义 网络结构 和 forward 的步骤
    - 继承nn.Module
    - __init__中实现一些层的定义
    - forward中实现推理的逻辑
- model：构建自己的 model，接口遵循Basemodel.py
    - （1）创建network(arch中创建的),__init__()加载预训练模型，并初始化训练相关的设置
    - （2）创建loss:init_training_settings(),调用build_loss()创建一个或者多个loss
    - （3）optimize_parameters：一次迭代下的train step，包含了network forward，loss计算，backward，和优化器的更新
    - （4）计算metric



# Rewrite code 
- degradation classifier
- degradation attention
- flux lora fintune code(modfiy from diffusers)# FLUX-Deg
