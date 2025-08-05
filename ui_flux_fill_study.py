# 导入gradio库，并使用别名gr。Gradio是一个非常方便的Python库，可以让你快速为你的机器学习模型、API或任何Python函数创建一个简单的Web用户界面。
import gradio as gr

# 导入subprocess库。这个库允许你在Python脚本中运行新的应用程序或命令。在这个脚本中，它被用来执行训练脚本。
import subprocess
# 导入json库。JSON是一种轻量级的数据交换格式。这个库用于处理配置文件（.json文件），比如保存和加载训练参数。
import json
# 导入sys库。它提供了访问由Python解释器使用或维护的变量和函数的功能。这里用它来获取Python解释器的路径，以确保用同一个解释器来运行子进程。
import sys
# 导入os库。这个库提供了很多与操作系统交互的功能，比如处理文件路径（检查文件是否存在、拼接路径等）。
import os

# ===== 语言翻译系统 =====
# 定义一个名为TRANSLATIONS的字典，用于存储多语言文本。这是一种实现国际化(i18n)的常见方法。
# 字典的键是语言代码（如 'zh' 代表中文, 'en' 代表英文）。
# 每个语言代码对应的值是另一个字典，其中包含了界面上所有需要翻译的文本。
TRANSLATIONS = {
    'zh': {
        'title': '## LoRA 训练',
        'script': '训练脚本',
        'config_path': '配置文件路径 (.json文件)',
        'config_path_placeholder': '输入保存/加载配置的路径',
        'save': '保存',
        'load': '加载',
        'directory_section': '目录配置',
        'output_dir': '输出目录',
        'output_dir_placeholder': '检查点保存位置',
        'save_name': '保存名称',
        'save_name_placeholder': '检查点保存名称',
        'pretrained_model_name_or_path': '预训练模型名称或路径',
        'pretrained_model_placeholder': '仓库名称或包含diffusers模型结构的目录',
        'resume_from_checkpoint': '从检查点恢复',
        'resume_checkpoint_placeholder': '从选定目录恢复lora权重',
        'train_data_dir': '训练数据目录',
        'train_data_dir_placeholder': '包含数据集的目录',
        'model_path': '模型路径',
        'model_path_placeholder': '如果不是从官方权重训练则为单个权重文件',
        'report_to': '报告到',
        'lora_config': 'LoRA 配置',
        'rank': '秩',
        'rank_info': '建议对小于100的训练集使用秩4',
        'train_batch_size': '训练批次大小',
        'batch_size_info': '批次大小1使用18GB。请使用小批次大小以避免内存不足',
        'repeats': '重复次数',
        'gradient_accumulation_steps': '梯度累积步数',
        'mixed_precision': '混合精度',
        'gradient_checkpointing': '梯度检查点',
        'optimizer': '优化器',
        'lr_scheduler': '学习率调度器',
        'cosine_restarts': '余弦重启',
        'cosine_restarts_info': '仅对学习率调度器cosine_with_restarts有用',
        'learning_rate': '学习率',
        'learning_rate_info': '推荐：1e-4 或 prodigy使用1',
        'lr_warmup_steps': '学习率预热步数',
        'seed': '随机种子',
        'blocks_to_swap': '交换块数',
        'blocks_to_swap_info': '交换到CPU的块数。建议24GB使用10，更低显存使用更多',
        'mask_dropout': '掩码丢弃',
        'mask_dropout_info': '丢弃掩码，意味着整个图像重建的掩码全为1',
        'reg_ratio': '正则化比率',
        'reg_ratio_info': '作为目标迁移学习的正则化。如果不训练不同目标则设为1',
        'reg_timestep': '正则化时间步',
        'reg_timestep_info': '作为目标迁移学习的正则化。如果不训练不同目标则设为0',
        'misc': '杂项',
        'num_train_epochs': '训练轮数',
        'num_train_epochs_info': '训练的总轮数',
        'save_model_epochs': '保存模型轮数',
        'save_model_epochs_info': '每x轮保存检查点',
        'validation_epochs': '验证轮数',
        'validation_epochs_info': '每x轮执行验证',
        'skip_epoch': '跳过轮数',
        'skip_epoch_info': '跳过x轮进行验证和保存检查点',
        'skip_step': '跳过步数',
        'skip_step_info': '跳过x步进行验证和保存检查点',
        'validation_ratio': '验证比例',
        'validation_ratio_info': '按此比例分割数据集用于验证',
        'recreate_cache': '重新创建缓存',
        'caption_dropout': '标题丢弃',
        'caption_dropout_info': '标题丢弃',
        'max_time_steps': '最大时间步限制',
        'max_time_steps_info': '最大时间步限制',
        'resolution_section': '## 实验选项：分辨率\n- 基于目标分辨率（默认：1024）。\n- 支持512或1024。',
        'resolution': '分辨率',
        'output_box': '输出框',
        'run': '运行',
        'language_toggle': '🌐 切换到English',
        
        'slider': '滑块训练相关',
        'use_two_captions': '使用两对应文本标注',
        'slider_positive_scale': '滑块正向目标强度',
        'slider_negative_scale': '滑块负面目标强度'
    },
    'en': {
        'title': '## Lora Training',
        'script': 'script',
        'config_path': 'Config Path (.json file)',
        'config_path_placeholder': 'Enter path to save/load config',
        'save': 'Save',
        'load': 'load',
        'directory_section': 'Directory section',
        'output_dir': 'output_dir',
        'output_dir_placeholder': 'checkpoint save to',
        'save_name': 'save_name',
        'save_name_placeholder': 'checkpoint save name',
        'pretrained_model_name_or_path': 'pretrained_model_name_or_path',
        'pretrained_model_placeholder': 'repo name or dir contains diffusers model structure',
        'resume_from_checkpoint': 'resume_from_checkpoint',
        'resume_checkpoint_placeholder': 'resume the lora weight from seleted dir',
        'train_data_dir': 'train_data_dir',
        'train_data_dir_placeholder': 'dir contains dataset',
        'model_path': 'model_path',
        'model_path_placeholder': 'single weight files if not trained from official weight',
        'report_to': 'report_to',
        'lora_config': 'Lora Config',
        'rank': 'rank',
        'rank_info': 'Recommanded to use rank 4 for training set small than 100.',
        'train_batch_size': 'train_batch_size',
        'batch_size_info': 'Batch size 1 is using 18GB. Please use small batch size to avoid oom.',
        'repeats': 'repeats',
        'gradient_accumulation_steps': 'gradient_accumulation_steps',
        'mixed_precision': 'mixed_precision',
        'gradient_checkpointing': 'gradient_checkpointing',
        'optimizer': 'optimizer',
        'lr_scheduler': 'lr_scheduler',
        'cosine_restarts': 'cosine_restarts',
        'cosine_restarts_info': 'Only useful for lr_scheduler: cosine_with_restarts',
        'learning_rate': 'learning_rate',
        'learning_rate_info': 'Recommended: 1e-4 or 1 for prodigy',
        'lr_warmup_steps': 'lr_warmup_steps',
        'seed': 'seed',
        'blocks_to_swap': 'blocks_to_swap',
        'blocks_to_swap_info': 'How many blocks to swap to cpu. It is suggested 10 for 24 GB and more for lower VRAM',
        'mask_dropout': 'mask_dropout',
        'mask_dropout_info': 'Dropout mask which means mask is all one for whole image reconstruction',
        'reg_ratio': 'reg_ratio',
        'reg_ratio_info': 'As regularization of objective transfer learning. Set as 1 if you aren\'t training different objective.',
        'reg_timestep': 'reg_timestep',
        'reg_timestep_info': 'As regularization of objective transfer learning. Set as 0 if you aren\'t training different objective.',
        'misc': 'Misc',
        'num_train_epochs': 'num_train_epochs',
        'num_train_epochs_info': 'Total epoches of the training',
        'save_model_epochs': 'save_model_epochs',
        'save_model_epochs_info': 'Save checkpoint when x epoches',
        'validation_epochs': 'validation_epochs',
        'validation_epochs_info': 'perform validation when x epoches',
        'skip_epoch': 'skip_epoch',
        'skip_epoch_info': 'Skip x epoches for validation and save checkpoint',
        'skip_step': 'skip_step',
        'skip_step_info': 'Skip x steps for validation and save checkpoint',
        'validation_ratio': 'validation_ratio',
        'validation_ratio_info': 'Split dataset with this ratio for validation',
        'recreate_cache': 'recreate_cache',
        'caption_dropout': 'Caption Dropout',
        'caption_dropout_info': 'Caption Dropout',
        'max_time_steps': 'Max timesteps limitation',
        'max_time_steps_info': 'Max timesteps limitation',
        'resolution_section': '## Experiment Option: resolution\n- Based target resolution (default:1024). \n- 512 or 1024 are supported.',
        'resolution': 'resolution',
        'output_box': 'Output Box',
        'run': 'Run',
        'language_toggle': '🌐 切换到中文',
        
        
        'slider': 'Slider Related',
        'use_two_captions': 'Use two captions for each direction',
        'slider_positive_scale': 'Slider positive scale',
        'slider_negative_scale': 'Slider negative scale'
    }
}

# 定义一个全局变量来存储当前的语言状态。
current_language = 'en'  # 默认设置为英文 'en'。你可以改为'zh'来默认显示中文。

def get_text(key):
    """
    定义一个函数，用于根据当前的语言设置，从TRANSLATIONS字典中获取对应的文本。
    :param key: 想要获取文本的键（例如 'title'）。
    :return: 返回当前语言对应的文本字符串。如果找不到，则返回键本身。
    """
    return TRANSLATIONS[current_language].get(key, key)

def toggle_language():
    """
    定义一个函数，用于切换语言状态。
    """
    # 声明我们将要修改的是全局变量 current_language。
    global current_language
    # 这是一个三元运算符，如果当前语言是'zh'，就切换到'en'；否则，切换到'zh'。
    current_language = 'en' if current_language == 'zh' else 'zh'
    # 返回切换后的语言代码。
    return current_language

# 定义一个名为 `default_config` 的字典，它存储了所有训练参数的默认值。
# 这对于首次启动程序或重置设置非常有用。
# 深度学习新手注意：这些参数就是所谓的“超参数”，调整它们会直接影响模型的训练效果和速度。
default_config = {
    "script": "train_flux_lora_ui_kontext.py", # 要执行的训练脚本文件名
    "script_choices": [ # 可供选择的训练脚本列表
                        "train_flux_lora_ui_kontext.py",
                        "train_flux_lora_ui_kontext_slider.py",
                        "train_flux_lora_ui_with_mask.py",
                        "train_flux_lora_ui.py",
                       ],
    "output_dir":"F:/models/flux", # 训练好的模型（检查点）保存的目录
    "save_name":"flux-lora", # 保存的模型文件的名字
    "pretrained_model_name_or_path":"F:/T2ITrainer/flux_models/kontext", # 基础模型路径，LoRA是在这个模型上进行微调的
    "train_data_dir":"F:/ImageSet/kontext", # 包含训练图片和标注的文件夹路径
    "resume_from_checkpoint":None, # 如果想从一个之前的训练检查点继续训练，在这里指定路径
    "model_path":None, # 如果基础模型不是官方的diffusers格式，而是一个单独的文件（如.safetensors），在这里指定
    "report_to":"all", # 训练日志报告给哪些平台，'all'通常指wandb和tensorboard
    "rank":16, # LoRA的秩。这是LoRA最重要的参数之一。越小，训练越快，模型文件越小，但可能学不到足够多的细节。越大反之。
    "train_batch_size":1, # 批次大小。一次训练迭代中处理的图片数量。受显存大小限制，越大通常效果越好但越耗显存。
    "repeats":1, # 数据集重复次数。每个epoch中，每张图片被重复训练的次数。
    "gradient_accumulation_steps":1, # 梯度累积步数。可以模拟更大的批次大小，当显存不足以设置大的train_batch_size时很有用。
    "mixed_precision":"bf16", # 混合精度训练。使用如bf16这样的低精度浮点数可以显著加快训练速度并减少显存占用。
    "gradient_checkpointing":True, # 梯度检查点。一种用计算时间换取显存空间的技术，可以让你用更少的显存训练更大的模型。
    "optimizer":"adamw", # 优化器。决定如何根据梯度更新模型权重的算法。AdamW是目前常用的选择。
    "lr_scheduler":"constant", # 学习率调度器。在训练过程中动态调整学习率的策略。
    "learning_rate":1e-4, # 学习率。这是训练中最重要的超参数之一，决定了每次更新权重时的步长。太高可能导致不稳定，太低可能导致训练过慢。
    "lr_warmup_steps":0, # 学习率预热步数。在训练开始时，学习率从0慢慢增长到设定值，有助于训练初期的稳定。
    "seed":4321, # 随机种子。固定这个值可以确保每次训练的结果都是可复现的。
    "num_train_epochs":5, # 训练轮数。一个epoch代表整个训练数据集被完整地过了一遍。
    "save_model_epochs":1, # 每隔多少个epoch保存一次模型检查点。
    "validation_epochs":1, # 每隔多少个epoch进行一次验证。
    "skip_epoch":0, # 跳过前N个epoch，不保存也不验证。
    "skip_step":0, # 跳过前N个step，不保存也不验证。
    "validation_ratio":0.1, # 从训练数据集中分出10%作为验证集。
    "recreate_cache":False, # 是否重新创建数据缓存。如果你的数据集有变动，需要设为True。
    "caption_dropout":0.1, # 标题丢弃率。训练时有10%的概率忽略图片的文字描述，强迫模型更多地从图像本身学习，是一种正则化手段。
    "config_path":"config.json", # 默认的配置文件名
    "resolution":"512", # 训练时图像的分辨率。
    "resolution_choices":["1024","768","512","256"], # 可选的分辨率列表。
    "use_debias":False, # 是否使用debias技术。
    "snr_gamma":0, # 信噪比gamma值，用于一种高级的训练策略。
    "cosine_restarts":1, # 当学习率调度器为cosine_with_restarts时，重启的次数。
    "max_time_steps":0, # 限制训练的最大时间步长。
    "blocks_to_swap":0, # 为了节省显存，将模型中的一些块交换到CPU内存。
    "mask_dropout":0, # 掩码丢弃率。
    "reg_ratio":0.0, # 正则化比率，用于目标迁移学习。
    "reg_timestep":0, # 正则化时间步，用于目标迁移学习。
    'use_two_captions': False, # 是否为滑块（slider）训练使用两种标题。
    'slider_positive_scale': 1.0, # 滑块训练的正向强度。
    'slider_negative_scale': -1.0 # 滑块训练的负向强度。
}


# 定义一个函数，用于将界面上的配置保存到JSON文件中。
# 这个函数接收大量参数，这些参数都对应UI界面的一个输入控件。
def save_config( 
        config_path,
        script,
        seed,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        recreate_cache,
        resolution,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        blocks_to_swap,
        mask_dropout,
        reg_ratio,
        reg_timestep,
        use_two_captions,
        slider_positive_scale,
        slider_negative_scale
    ):
    # 将所有传入的参数收集到一个字典中，这个字典的结构和 `default_config` 保持一致。
    config = {
        "script":script,
        "seed":seed,
        "mixed_precision":mixed_precision,
        "report_to":report_to,
        "lr_warmup_steps":lr_warmup_steps,
        "output_dir":output_dir,
        "save_name":save_name,
        "train_data_dir":train_data_dir,
        "optimizer":optimizer,
        "lr_scheduler":lr_scheduler,
        "learning_rate":learning_rate,
        "train_batch_size":train_batch_size,
        "repeats":repeats,
        "gradient_accumulation_steps":gradient_accumulation_steps,
        "num_train_epochs":num_train_epochs,
        "save_model_epochs":save_model_epochs,
        "validation_epochs":validation_epochs,
        "rank":rank,
        "skip_epoch":skip_epoch,
        "skip_step":skip_step,
        "gradient_checkpointing":gradient_checkpointing,
        "validation_ratio":validation_ratio,
        "pretrained_model_name_or_path":pretrained_model_name_or_path,
        "model_path":model_path,
        "resume_from_checkpoint":resume_from_checkpoint,
        "recreate_cache":recreate_cache,
        "config_path":config_path,
        "resolution":resolution,
        "caption_dropout":caption_dropout,
        "cosine_restarts":cosine_restarts,
        "max_time_steps":max_time_steps,
        "blocks_to_swap":blocks_to_swap,
        "mask_dropout":mask_dropout,
        "reg_ratio":reg_ratio,
        "reg_timestep":reg_timestep,
        'use_two_captions': use_two_captions,
        'slider_positive_scale': slider_positive_scale,
        'slider_negative_scale': slider_negative_scale
    }
    # 使用 'with open' 语句来打开一个文件。'w' 表示写入模式。这种方式可以确保文件在使用后被正确关闭。
    with open(config_path, 'w') as f:
        # 使用 json.dump() 函数将 `config` 字典写入到文件中。
        # indent=4 参数会让JSON文件格式化得更美观，易于阅读。
        json.dump(config, f, indent=4)
    # 在控制台打印一条消息，告诉用户配置已保存。
    print(f"Configuration saved to {config_path}")
    print(f"Update default config")
    # 同时，也将这份最新的配置保存为 "config.json"，作为下一次启动时的默认配置。
    with open("config.json", 'w') as f:
        json.dump(config, f, indent=4)

# 定义一个函数，用于从JSON文件中加载配置。
def load_config(config_path):
    # 检查文件名是否以 ".json" 结尾。
    if not config_path.endswith(".json"):
        print("!!!File is not json format.")
        print("Load default config")
        # 如果不是，就使用默认的 "config.json" 文件。
        config_path = "config.json"
    # 检查指定的配置文件路径是否存在。
    if not os.path.exists(config_path):
        # 如果文件不存在，就创建一个新的。
        with open(config_path, 'w') as f:
            config = {}
            # 从 `default_config` 字典中读取所有默认设置。
            for key in default_config.keys():
                config[key] = default_config[key]
            # 将默认配置写入新创建的文件中。
            json.dump(config, f, indent=4)
        # 返回默认配置。
        return config
    # 拼接完整的文件路径（虽然在这里可能不是必须的，但这是一个好习惯）。
    config_path = os.path.join(config_path)
    try:
        # 使用 'with open' 尝试打开并读取JSON文件。'r' 表示读取模式。
        with open(config_path, 'r') as f:
            # 使用 json.load() 函数从文件中加载数据并解析为Python字典。
            config = json.load(f)
    except:
        # 如果在读取或解析过程中发生任何错误（比如文件损坏），则加载默认的 "config.json"。
        config_path = "config.json"
    # 在控制台打印消息，告知用户配置已加载。
    print(f"Loaded configuration from {config_path}")
    # 遍历加载到的配置字典中的所有键。
    for key in config.keys():
        # 用加载到的值更新 `default_config` 字典。这样可以确保UI显示的是加载的配置。
        default_config[key] = config[key]
            
    # 这个函数最关键的一步：返回一个长长的元组(tuple)，包含了所有从配置文件中读取到的值。
    # 这些返回值将按顺序传递给Gradio界面中对应的输出组件，从而更新整个UI。
    return config_path,default_config['script'],default_config['seed'], \
           default_config['mixed_precision'],default_config['report_to'],default_config['lr_warmup_steps'], \
           default_config['output_dir'],default_config['save_name'],default_config['train_data_dir'], \
           default_config['optimizer'],default_config['lr_scheduler'],default_config['learning_rate'], \
           default_config['train_batch_size'],default_config['repeats'],default_config['gradient_accumulation_steps'], \
           default_config['num_train_epochs'],default_config['save_model_epochs'],default_config['validation_epochs'], \
           default_config['rank'],default_config['skip_epoch'], \
           default_config['skip_step'],default_config['gradient_checkpointing'],default_config['validation_ratio'], \
           default_config['pretrained_model_name_or_path'],default_config['model_path'],default_config['resume_from_checkpoint'], \
           default_config['recreate_cache'],default_config['resolution'], \
           default_config['caption_dropout'], \
           default_config['cosine_restarts'],default_config['max_time_steps'], \
           default_config['blocks_to_swap'],default_config['mask_dropout'], \
           default_config['reg_ratio'],default_config['reg_timestep'], \
           default_config['use_two_captions'],default_config['slider_positive_scale'],default_config['slider_negative_scale']


# 在脚本启动时，默认加载一次 "config.json" 文件，初始化界面设置。
load_config("config.json")
# 定义 "运行" 按钮被点击时要执行的核心函数。
def run(
        config_path,
        script,
        seed,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        recreate_cache,
        resolution,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        blocks_to_swap,
        mask_dropout,
        reg_ratio,
        reg_timestep,
        use_two_captions,
        slider_positive_scale,
        slider_negative_scale
    ):
    # 第一步：调用 `save_config` 函数，将当前界面上所有的设置都保存到指定的 `config_path` 文件中。
    # 这样做是为了将所有参数打包到一个文件中，方便传递给训练脚本。
    save_config(
        config_path,
        script,
        seed,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        recreate_cache,
        resolution,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        blocks_to_swap,
        mask_dropout,
        reg_ratio,
        reg_timestep,
        use_two_captions,
        slider_positive_scale,
        slider_negative_scale
    )

    # 第二步：构建要执行的命令行指令。
    # 这是一个列表，包含了命令的各个部分。
    # sys.executable 指的是当前运行这个UI脚本的Python解释器的路径。
    # script 是从UI界面选择的训练脚本文件名。
    # "--config_path" 是传递给训练脚本的命令行参数名。
    # config_path 是包含所有配置的json文件的路径。
    # 这种方式（只传递一个配置文件）比把几十个参数全部列在命令行里要整洁和可靠得多。
    command_args = [sys.executable, script, "--config_path", config_path]

    # 第三步：使用 `subprocess.call()` 执行上面构建好的命令。
    # 这会启动一个新的进程来运行训练脚本，而不会阻塞当前的Gradio界面。
    subprocess.call(command_args)

    # 第四步：将执行的命令拼接成一个字符串并返回。
    # 这个字符串会显示在UI的输出框中，让用户知道后台实际执行了什么命令。
    return " ".join(command_args)




# 定义语言切换按钮的事件处理函数。
def toggle_language_handler():
    """语言切换处理函数 (这个函数在后面的代码中没有被直接使用，而是被更完整的 update_language_interface 替代了)"""
    # 调用函数切换全局语言状态。
    toggle_language()
    # 创建一个空列表，用于存放需要更新的Gradio组件。
    updates = []
    # 更新标题组件。
    updates.append(gr.Markdown(get_text('title')))
    # 更新语言切换按钮的文本。
    updates.append(gr.Button(get_text('language_toggle'), scale=0, size="sm"))
    # 返回包含所有更新后组件的列表。Gradio会根据这个列表来更新界面。
    return updates

# 使用 `with gr.Blocks() as demo:` 来创建一个Gradio界面。所有UI组件都将定义在这个代码块内。
with gr.Blocks() as demo:
    # 使用 `gr.Row()` 创建一个水平布局的行。
    with gr.Row():
        # 这是一个小技巧，用一个HTML div来占据所有可用空间，从而把后面的按钮推到最右边。
        gr.HTML("<div style='flex-grow: 1;'></div>")  # 占位符，让按钮右对齐
        # 创建语言切换按钮。`scale=0` 表示它不随窗口缩放，`size="sm"` 表示小尺寸。
        language_toggle_btn = gr.Button(get_text('language_toggle'), scale=0, size="sm")
    
    # 创建一个Markdown组件来显示标题。`get_text`确保了标题是当前选定语言的。
    title_md = gr.Markdown(get_text('title'))
    
    # 创建一个下拉菜单，用于选择要运行的训练脚本。
    # label是标签文本, value是默认值, choices是可选项列表。
    script = gr.Dropdown(label=get_text('script'), value=default_config["script"], choices=default_config["script_choices"])
    
    # 创建一个新行，`equal_height=True` 确保行内所有组件高度一致。
    with gr.Row(equal_height=True):
        # 创建一个文本输入框，用于输入配置文件的路径。
        # `scale=3`表示它在行内占据的相对宽度是3。
        # `placeholder`是输入框为空时显示的提示文本。
        config_path = gr.Textbox(scale=3, label=get_text('config_path'), value=default_config["config_path"], placeholder=get_text('config_path_placeholder'))
        # 创建“保存”按钮。
        save_config_btn = gr.Button(get_text('save'), scale=1)
        # 创建“加载”按钮。
        load_config_btn = gr.Button(get_text('load'), scale=1)

    # 创建一个可折叠的区域（Accordion），用于组织目录相关的设置。
    directory_accordion = gr.Accordion(get_text('directory_section'))
    with directory_accordion:
        # 在折叠区域内创建一行。
        with gr.Row():
            # 输出目录输入框
            output_dir = gr.Textbox(label=get_text('output_dir'), value=default_config["output_dir"],
                                      placeholder=get_text('output_dir_placeholder'))
            # 保存名称输入框
            save_name = gr.Textbox(label=get_text('save_name'), value=default_config["save_name"],
                                      placeholder=get_text('save_name_placeholder'))
        with gr.Row():
            # 预训练模型路径输入框
            pretrained_model_name_or_path = gr.Textbox(label=get_text('pretrained_model_name_or_path'), 
                value=default_config["pretrained_model_name_or_path"], 
                placeholder=get_text('pretrained_model_name_or_path_placeholder')
            )
            # 从检查点恢复路径输入框
            resume_from_checkpoint = gr.Textbox(label=get_text('resume_from_checkpoint'), value=default_config["resume_from_checkpoint"], placeholder=get_text('resume_from_checkpoint_placeholder'))
        with gr.Row():
            # 训练数据目录输入框
            train_data_dir = gr.Textbox(label=get_text('train_data_dir'), value=default_config["train_data_dir"], placeholder=get_text('train_data_dir_placeholder'))
            # 单独模型文件路径输入框
            model_path = gr.Textbox(label=get_text('model_path'), value=default_config["model_path"], placeholder=get_text('model_path_placeholder'))
        with gr.Row():
            # 报告目标下拉菜单
            report_to = gr.Dropdown(label=get_text('report_to'), value=default_config["report_to"], choices=["all","wandb","tensorboard"])

    # 创建另一个折叠区域，用于LoRA训练的核心参数配置。
    lora_accordion = gr.Accordion(get_text('lora_config'))
    with lora_accordion:
        # 在折叠区域内创建多行来布局各个参数控件。
        with gr.Row():
            # Rank (秩) 输入框，类型为数字。`info`参数会在标签旁边显示一段帮助信息。
            rank = gr.Number(label=get_text('rank'), value=default_config["rank"], info=get_text('rank_info'))
            # 训练批次大小输入框
            train_batch_size = gr.Number(label=get_text('train_batch_size'), value=default_config["train_batch_size"], info=get_text('train_batch_size_info'))
        with gr.Row():
            # 重复次数输入框
            repeats = gr.Number(label=get_text('repeats'), value=default_config["repeats"])
            # 梯度累积步数输入框
            gradient_accumulation_steps = gr.Number(label=get_text('gradient_accumulation_steps'), value=default_config["gradient_accumulation_steps"])
            # 混合精度选择，使用单选按钮（Radio）。
            mixed_precision = gr.Radio(label=get_text('mixed_precision'), value=default_config["mixed_precision"], choices=["bf16", "fp8"])
            # 梯度检查点，使用复选框（Checkbox）。
            gradient_checkpointing = gr.Checkbox(label=get_text('gradient_checkpointing'), value=default_config["gradient_checkpointing"])
        with gr.Row():
            # 优化器下拉菜单
            optimizer = gr.Dropdown(label=get_text('optimizer'), value=default_config["optimizer"], choices=["adamw","prodigy"])
            # 学习率调度器下拉菜单
            lr_scheduler = gr.Dropdown(label=get_text('lr_scheduler'), value=default_config["lr_scheduler"], 
                                       choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"])
            # 余弦重启次数输入框，`minimum=1` 限制了最小值为1。
            cosine_restarts = gr.Number(label=get_text('cosine_restarts'), value=default_config["cosine_restarts"], info=get_text('cosine_restarts_info'), minimum=1)
        with gr.Row():
            # 学习率输入框
            learning_rate = gr.Number(label=get_text('learning_rate'), value=default_config["learning_rate"], info=get_text('learning_rate_info'))
            # 学习率预热步数输入框
            lr_warmup_steps = gr.Number(label=get_text('lr_warmup_steps'), value=default_config["lr_warmup_steps"])
            # 随机种子输入框
            seed = gr.Number(label=get_text('seed'), value=default_config["seed"])
        with gr.Row():
            # 交换块数输入框
            blocks_to_swap = gr.Number(label=get_text('blocks_to_swap'), value=default_config["blocks_to_swap"], info=get_text('blocks_to_swap_info'))
            # 掩码丢弃率输入框
            mask_dropout = gr.Number(label=get_text('mask_dropout'), value=default_config["mask_dropout"], info=get_text('mask_dropout_info'))
            # 正则化比率输入框
            reg_ratio = gr.Number(label=get_text('reg_ratio'), value=default_config["reg_ratio"], info=get_text('reg_ratio_info'))
            # 正则化时间步输入框
            reg_timestep = gr.Number(label=get_text('reg_timestep'), value=default_config["reg_timestep"], info=get_text('reg_timestep_info'))
            
            
    # 创建第三个折叠区域，用于一些杂项设置。
    misc_accordion = gr.Accordion(get_text('misc'))
    with misc_accordion:
        with gr.Row():
            # 训练轮数输入框
            num_train_epochs = gr.Number(label=get_text('num_train_epochs'), value=default_config["num_train_epochs"], info=get_text('num_train_epochs_info'))
            # 保存模型轮数输入框
            save_model_epochs = gr.Number(label=get_text('save_model_epochs'), value=default_config["save_model_epochs"], info=get_text('save_model_epochs_info'))
            # 验证轮数输入框
            validation_epochs = gr.Number(label=get_text('validation_epochs'), value=default_config["validation_epochs"], info=get_text('validation_epochs_info'))
        with gr.Row():
            # 跳过轮数输入框
            skip_epoch = gr.Number(label=get_text('skip_epoch'), value=default_config["skip_epoch"], info=get_text('skip_epoch_info'))
            # 跳过步数输入框
            skip_step = gr.Number(label=get_text('skip_step'), value=default_config["skip_step"], info=get_text('skip_step_info'))
            # 验证集比例输入框
            validation_ratio = gr.Number(label=get_text('validation_ratio'), value=default_config["validation_ratio"], info=get_text('validation_ratio_info'))
            
        with gr.Row():
            # 重新创建缓存复选框
            recreate_cache = gr.Checkbox(label=get_text('recreate_cache'), value=default_config["recreate_cache"])
            # 标题丢弃率输入框, `maximum=1, minimum=0` 限制了取值范围在0到1之间。
            caption_dropout = gr.Number(label=get_text('caption_dropout'), value=default_config["caption_dropout"], info=get_text('caption_dropout_info'), maximum=1, minimum=0)
            # 最大时间步输入框
            max_time_steps = gr.Number(label=get_text('max_time_steps'), value=default_config["max_time_steps"], info=get_text('max_time_steps_info'), maximum=1000, minimum=0)
        
        # 显示分辨率相关的说明文本
        resolution_md = gr.Markdown(get_text('resolution_section'))
        with gr.Row():
            # 分辨率选择下拉菜单
            resolution = gr.Dropdown(label=get_text('resolution'), value=default_config["resolution"], choices=default_config["resolution_choices"])
    
    # 创建第四个折叠区域，用于滑块（Slider）训练相关的特殊参数。
    misc_accordion = gr.Accordion(get_text('slider'))
    with misc_accordion:
        with gr.Row():
            # 是否使用两种标题的复选框
            use_two_captions = gr.Checkbox(label=get_text('use_two_captions'), value=default_config["use_two_captions"])
            # 滑块正向强度输入框
            slider_positive_scale = gr.Number(label=get_text('slider_positive_scale'), value=default_config["slider_positive_scale"])
            # 滑块负向强度输入框
            slider_negative_scale = gr.Number(label=get_text('slider_negative_scale'), value=default_config["slider_negative_scale"])
        
    
    
    # 创建一个文本框，用于显示输出信息（比如执行的命令）。
    output = gr.Textbox(label=get_text('output_box'))
    # 创建“运行”按钮。
    run_btn = gr.Button(get_text('run'))
    
    # 定义一个列表，包含了所有需要作为函数输入的UI组件。
    # 这个列表的顺序非常重要，必须和 `run`、`save_config` 等函数的参数顺序一一对应。
    inputs = [
        config_path,
        script,
        seed,
        mixed_precision,
        report_to,
        lr_warmup_steps,
        output_dir,
        save_name,
        train_data_dir,
        optimizer,
        lr_scheduler,
        learning_rate,
        train_batch_size,
        repeats,
        gradient_accumulation_steps,
        num_train_epochs,
        save_model_epochs,
        validation_epochs,
        rank,
        skip_epoch,
        skip_step,
        gradient_checkpointing,
        validation_ratio,
        pretrained_model_name_or_path,
        model_path,
        resume_from_checkpoint,
        recreate_cache,
        resolution,
        caption_dropout,
        cosine_restarts,
        max_time_steps,
        blocks_to_swap,
        mask_dropout,
        reg_ratio,
        reg_timestep,
        use_two_captions,
        slider_positive_scale,
        slider_negative_scale
    ]
    
    # 定义一个完整的语言更新函数。
    def update_language_interface():
        """更新界面语言，返回所有需要更新的组件的新实例"""
        # 切换语言状态
        toggle_language()
        # 创建一个列表，包含所有UI组件的“更新后”版本。
        # Gradio会用这个列表中的新组件替换掉旧组件。
        # 注意：这里是通过创建一个全新的组件实例（例如 `gr.Markdown(...)`）来实现更新的。
        updated_components = [
            # 基础组件
            gr.Markdown(value=get_text('title')),  # 标题 (更新时用 value)
            gr.Button(value=get_text('language_toggle')),  # 语言切换按钮 (更新时用 value)
            gr.Dropdown(label=get_text('script')),  # 脚本选择 (只更新 label)
            gr.Textbox(label=get_text('config_path'), placeholder=get_text('config_path_placeholder')),  # 配置路径
            gr.Button(value=get_text('save')),  # 保存按钮
            gr.Button(value=get_text('load')),  # 加载按钮
            
            # Accordion组件更新 (只更新 label)
            gr.Accordion(label=get_text('directory_section')),  # 目录配置标题
            gr.Accordion(label=get_text('lora_config')),  # LoRA设置标题
            gr.Accordion(label=get_text('misc')),  # 杂项标题
            
            # 目录设置部分的组件
            gr.Textbox(label=get_text('output_dir'), placeholder=get_text('output_dir_placeholder')),  # 输出目录
            gr.Textbox(label=get_text('save_name'), placeholder=get_text('save_name_placeholder')),  # 保存名称
            gr.Textbox(label=get_text('pretrained_model_name_or_path'), placeholder=get_text('pretrained_model_name_or_path_placeholder')),  # 预训练模型路径
            gr.Textbox(label=get_text('resume_from_checkpoint'), placeholder=get_text('resume_from_checkpoint_placeholder')),  # 恢复检查点
            gr.Textbox(label=get_text('train_data_dir'), placeholder=get_text('train_data_dir_placeholder')),  # 训练数据目录
            gr.Textbox(label=get_text('model_path'), placeholder=get_text('model_path_placeholder')),  # 模型路径
            gr.Dropdown(label=get_text('report_to')),  # 报告到
            
            # LoRA配置部分的组件
            gr.Number(label=get_text('rank'), info=get_text('rank_info')),  # 秩
            gr.Number(label=get_text('train_batch_size'), info=get_text('batch_size_info')),  # 训练批次大小
            gr.Number(label=get_text('repeats')),  # 重复次数
            gr.Number(label=get_text('gradient_accumulation_steps')),  # 梯度累积步数
            gr.Radio(label=get_text('mixed_precision')),  # 混合精度
            gr.Checkbox(label=get_text('gradient_checkpointing')),  # 梯度检查点
            gr.Dropdown(label=get_text('optimizer')),  # 优化器
            gr.Dropdown(label=get_text('lr_scheduler')),  # 学习率调度器
            gr.Number(label=get_text('cosine_restarts'), info=get_text('cosine_restarts_info')),  # 余弦重启
            gr.Number(label=get_text('learning_rate'), info=get_text('learning_rate_info')),  # 学习率
            gr.Number(label=get_text('lr_warmup_steps')),  # 学习率预热步数
            gr.Number(label=get_text('seed')),  # 随机种子
            gr.Number(label=get_text('blocks_to_swap'), info=get_text('blocks_to_swap_info')),  # 交换块数
            gr.Number(label=get_text('mask_dropout'), info=get_text('mask_dropout_info')),  # 掩码丢弃
            gr.Number(label=get_text('reg_ratio'), info=get_text('reg_ratio_info')),  # 正则化比率
            gr.Number(label=get_text('reg_timestep'), info=get_text('reg_timestep_info')),  # 正则化时间步
            
            # Misc部分的组件
            gr.Number(label=get_text('num_train_epochs'), info=get_text('num_train_epochs_info')),  # 训练轮数
            gr.Number(label=get_text('save_model_epochs'), info=get_text('save_model_epochs_info')),  # 保存模型轮数
            gr.Number(label=get_text('validation_epochs'), info=get_text('validation_epochs_info')),  # 验证轮数
            gr.Number(label=get_text('skip_epoch'), info=get_text('skip_epoch_info')),  # 跳过轮数
            gr.Number(label=get_text('skip_step'), info=get_text('skip_step_info')),  # 跳过步数
            gr.Number(label=get_text('validation_ratio'), info=get_text('validation_ratio_info')),  # 验证比率
            gr.Checkbox(label=get_text('recreate_cache')),  # 重建缓存
            gr.Number(label=get_text('caption_dropout'), info=get_text('caption_dropout_info')),  # 标题丢弃
            gr.Number(label=get_text('max_time_steps'), info=get_text('max_time_steps_info')),  # 最大时间步
            gr.Markdown(value=get_text('resolution_section')),  # 分辨率说明
            gr.Dropdown(label=get_text('resolution')),  # 分辨率
            
            # 输出和运行按钮
            gr.Textbox(label=get_text('output_box')),  # 输出框
            gr.Button(value=get_text('run')),  # 运行按钮 (这里应该是 run, 不是 run_button)
            
            # Slider相关
            gr.Checkbox(label=get_text('use_two_captions')),
            gr.Number(label=get_text('slider_positive_scale')),
            gr.Number(label=get_text('slider_negative_scale'))
        ]
        return updated_components
    
    # 绑定事件处理器。这是Gradio的核心机制，将用户的操作（如点击按钮）与Python函数关联起来。
    # 当 "运行" 按钮被点击时，执行 `run` 函数。
    # `inputs` 指定了哪些UI组件的值会作为参数传给 `run` 函数。
    # `outputs` 指定了 `run` 函数的返回值应该更新哪个UI组件。
    # `api_name` 使得这个功能可以通过API被调用。
    run_btn.click(fn=run, inputs=inputs, outputs=output, api_name="run")
    # 当 "保存" 按钮被点击时，执行 `save_config` 函数。
    save_config_btn.click(fn=save_config, inputs=inputs)
    # 当 "加载" 按钮被点击时，执行 `load_config` 函数。
    # 它的输入只有 `config_path` 文本框。
    # 它的输出是 `inputs` 列表中的所有组件，因为 `load_config` 函数会返回所有加载到的值来更新整个界面。
    load_config_btn.click(fn=load_config, inputs=[config_path], outputs=inputs)
    
    # 语言切换事件处理。
    # 当 `language_toggle_btn` 被点击时，执行 `update_language_interface` 函数。
    # 这个函数没有输入 (`inputs=[]`)。
    # 它的输出是一个非常长的列表，包含了界面上几乎所有的组件。
    # 这是因为切换语言需要更新所有显示文本的组件的 `label`, `value`, `placeholder` 或 `info` 属性。
    language_toggle_btn.click(
        fn=update_language_interface,
        inputs=[],
        # 这个列表里的组件会接收 `update_language_interface` 函数返回的新组件实例，从而完成界面更新。
        # 这里的组件顺序需要和 `update_language_interface` 函数返回的列表顺序严格对应。
        outputs=[
            title_md, language_toggle_btn, script, config_path, save_config_btn, load_config_btn,
            directory_accordion, lora_accordion, misc_accordion,  # 添加Accordion组件
            output_dir, save_name, pretrained_model_name_or_path, resume_from_checkpoint, 
            train_data_dir, model_path, report_to,
            rank, train_batch_size, repeats, gradient_accumulation_steps, mixed_precision, gradient_checkpointing,
            optimizer, lr_scheduler, cosine_restarts, learning_rate, lr_warmup_steps, seed,
            blocks_to_swap, mask_dropout, reg_ratio, reg_timestep,
            num_train_epochs, save_model_epochs, validation_epochs, skip_epoch, skip_step, validation_ratio,
            recreate_cache, caption_dropout, max_time_steps, resolution_md, resolution,
            output, run_btn, 
            # Slider相关的组件也需要在这里列出以被更新
            use_two_captions, slider_positive_scale, slider_negative_scale
        ]
    )

# 这是一个标准的Python入口点检查。
# `__name__ == "__main__"` 这个条件只在当前脚本被直接运行时才成立（而不是被其他脚本导入时）。
if __name__ == "__main__":
    # 调用 `demo.launch()` 来启动Gradio的Web服务器。
    # 这会生成一个本地URL，你可以在浏览器中打开它来访问和使用这个UI界面。
    demo.launch()