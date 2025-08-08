# train_refactored.py

from datetime import datetime
# import jsonlines
import copy
import safetensors
import argparse
# import functools
import gc
# import logging
import math
import os
import random
# import shutil
# from pathlib import Path

import accelerate
# import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
import diffusers

# from diffusers.image_processor import VaeImageProcessor

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from accelerate.logging import get_logger
import logging  # 确保导入了 logging
from accelerate.logging import get_logger
import os # 确保导入了 os
# from datasets import load_dataset
# from packaging import version
# from torchvision import transforms
# from torchvision.transforms.functional import crop

from tqdm.auto import tqdm
# from transformers import AutoTokenizer, PretrainedConfig
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    # FluxTransformer2DModel,
)

from flux.transformer_flux_masked import MaskedFluxTransformer2DModel
from flux.flux_utils import compute_loss_weighting_for_sd3, compute_density_for_timestep_sampling
from flux.pipeline_flux_kontext import FluxKontextPipeline

from pathlib import Path
from diffusers.optimization import get_scheduler
# from diffusers.training_utils import _set_state_dict_into_text_encoder, cast_training_params, compute_snr
from diffusers.training_utils import (
    cast_training_params,
    compute_snr
)
from diffusers.utils import (
    # check_min_version,
    convert_all_state_dict_to_peft,
    convert_state_dict_to_diffusers,
    convert_state_dict_to_kohya,
    convert_unet_state_dict_to_peft,
    # compute_density_for_timestep_sampling,
    is_wandb_available,
    # compute_loss_weighting_for_sd3,
)
from diffusers.loaders import LoraLoaderMixin
# from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from tqdm import tqdm
# from PIL import Image

# from sklearn.model_selection import train_test_split # 不再需要

from pathlib import Path
import json

# =========================================================================
# ======================== 新增的导入和全局配置 ============================
# =========================================================================
import yaml
import importlib

# 定义配置文件的路径
# 该YAML文件专门用于配置数据集的加载路径
CONFIG_YAML_PATH = "config.yaml"

# 定义用于所有训练和验证图像的固定文本提示
# 这将取代原有的、为每张图片读取标题文件的逻辑
FIXED_PROMPT = "Remove digital noise and grain, keep the composition unchanged"
# =========================================================================
# =========================================================================


# import sys
# from utils.image_utils_kolors import BucketBatchSampler, CachedImageDataset, create_metadata_cache
# from utils.image_utils_flux import CachedMutiImageDataset # 不再需要
# from utils.bucket.bucket_batch_sampler import BucketBatchSampler # 不再需要

# from prodigyopt import Prodigy

# https://github.com/Lightning-AI/pytorch-lightning/blob/0d52f4577310b5a1624bed4d23d49e37fb05af9e/src/lightning_fabric/utilities/seed.py
from random import getstate as python_get_rng_state
from random import setstate as python_set_rng_state

from peft import LoraConfig, prepare_model_for_kbit_training
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
# from kolors.models.modeling_chatglm import ChatGLMModel
# from kolors.models.tokenization_chatglm import ChatGLMTokenizer

from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast, BitsAndBytesConfig

if is_wandb_available():
    import wandb

from safetensors.torch import save_file

from utils.dist_utils import flush

from hashlib import md5
import glob
import shutil
from collections import defaultdict

# 保留原有的辅助函数，因为它们在某些情况下可能仍被内部逻辑使用
from utils.image_utils_flux import load_image, compute_text_embeddings, replace_non_utf8_characters, create_empty_embedding, get_empty_embedding, cache_file, cache_multiple, crop_image,get_md5_by_path,read_image

# from diffusers import FluxPriorReduxPipeline
import cv2

from torchvision import transforms

from diffusers.image_processor import VaeImageProcessor

from utils.utils import find_index_from_right, ToTensorUniversal


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder"
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2"
    )
    return text_encoder_one, text_encoder_two

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder
    )
    model_class = text_encoder_config.architectures[0]
    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel
        return CLIPTextModel
    elif model_class == "T5EncoderModel":
        from transformers import T5EncoderModel
        return T5EncoderModel
    else:
        raise ValueError(f"{model_class} is not supported.")

logger = get_logger(__name__)




def memory_stats():
    print("\nmemory_stats:\n")
    print(torch.cuda.memory_allocated()/1024**2)

# --- 参数解析函数保持不变 ---
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=False,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help=(
            "Run validation every X epochs."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="flux-dreambooth",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    
    parser.add_argument(
        "--save_name",
        type=str,
        default="flux_",
        help=(
            "save name prefix for saving checkpoints"
        ),
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )

    # parser.add_argument(
    #     "--scale_lr",
    #     action="store_true",
    #     default=False,
    #     help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    # )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--cosine_restarts",
        type=int,
        default=1,
        help=(
            'for lr_scheduler cosine_with_restarts'
        ),
    )
    
    
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=50, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
    )

    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
    )

    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
    )
    parser.add_argument(
        "--prodigy_beta3",
        type=float,
        default=None,
        help="coefficients for computing the Prodidy stepsize using running averages. If set to None, "
        "uses the value of square root of beta2. Ignored if optimizer is adamW",
    )
    parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-02, help="Weight decay to use for unet params")
    parser.add_argument(
        "--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
    )

    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
    )

    parser.add_argument(
        "--prodigy_use_bias_correction",
        type=bool,
        default=True,
        help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_safeguard_warmup",
        type=bool,
        default=True,
        help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
        "Ignored if optimizer is adamW",
    )
    parser.add_argument(
        "--prodigy_d_coef",
        type=float,
        default=2,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["bf16", "fp8"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="",
        help=(
            "train data image folder"
        ),
    )
    
    
    # parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument(
        "--save_model_epochs",
        type=int,
        default=1,
        help=("Save model when x epochs"),
    )
    parser.add_argument(
        "--save_model_steps",
        type=int,
        default=-1,
        help=("Save model when x steps"),
    )
    parser.add_argument(
        "--skip_epoch",
        type=int,
        default=0,
        help=("skip val and save model before x epochs"),
    )
    parser.add_argument(
        "--skip_step",
        type=int,
        default=0,
        help=("skip val and save model before x step"),
    )
    
    # parser.add_argument(
    #     "--break_epoch",
    #     type=int,
    #     default=0,
    #     help=("break training after x epochs"),
    # )
    parser.add_argument(
        "--validation_ratio",
        type=float,
        default=0.1,
        help=("dataset split ratio for validation"),
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help=("seperate model path"),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--recreate_cache",
        action="store_true",
        help="recreate all cache",
    )
    parser.add_argument(
        "--caption_dropout",
        type=float,
        default=0.1,
        help=("caption_dropout ratio which drop the caption and update the unconditional space"),
    )
    parser.add_argument(
        "--mask_dropout",
        type=float,
        default=0.01,
        help=("mask_dropout ratio which replace the mask with all 0"),
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help=("seperate vae path"),
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default='512',
        help=("default: '1024', accept str: '1024', '512'"),
    )
    parser.add_argument(
        "--use_debias",
        action="store_true",
        help="Use debiased estimation loss",
    )
    
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=5,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--max_time_steps",
        type=int,
        default=1000,
        help="Max time steps limitation. The training timesteps would limited as this value. 0 to max_time_steps",
    )
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="logit_normal",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "logit_snr"],
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--freeze_transformer_layers",
        type=str,
        default='',
        help="Stop training the transformer layers included in the input using ',' to seperate layers. Example: 5,7,10,17,18,19"
    )
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma seperated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1,
        help="the FLUX.1 dev variant is a guidance distilled model. default 1 to preserve distillation.",
    )
    # parser.add_argument(
    #     "--use_fp8",
    #     action="store_true",
    #     help="Use fp8 model",
    # )
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=10,
        help="Suggest to 10-20 depends on VRAM",
    )
    parser.add_argument(
        "--noise_offset",
        type=float,
        default=0.01,
        help="noise offset in initial noise",
    )
    parser.add_argument(
        "--reg_ratio",
        type=float,
        default=0.0,
        help="As regularization of objective transfer learning. Set as 1 if you aren't training different objective.",
    )
    parser.add_argument(
        "--reg_timestep",
        type=int,
        default=0,
        help="As regularization of objective transfer learning. You could try different value.",
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.json",
        help="Path to the config file.",
    )
    parser.add_argument(
        "--use_two_captions",
        action="store_true",
        help="Use _T caption and _R caption to train each direction",
    )
    parser.add_argument(
        "--slider_positive_scale",
        type=float,
        default=1.0,
        help="Slider Training positive target scale",
    )
    parser.add_argument(
        "--slider_negative_scale",
        type=float,
        default=-1.0,
        help="Slider Training negative target scale",
    )
    
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    # Load config file if provided
    if args.config_path and os.path.exists(args.config_path):
        try:
            with open(args.config_path, 'r', encoding='utf-8') as f:
                config_args = json.load(f)
            # Update args with values from config file
            # Ensure that config values override command-line arguments
            # Convert config values to the correct types if necessary
            for key, value in config_args.items():
                if hasattr(args, key):
                    # Attempt to convert value to the type of the existing argument
                    arg_type = type(value)
                    if arg_type == bool:
                        # Handle boolean conversion carefully
                        if isinstance(value, str):
                            if value.lower() in ('true', '1', 'yes'):
                                setattr(args, key, True)
                            elif value.lower() in ('false', '0', 'no'):
                                setattr(args, key, False)
                            else:
                                print(f"Could not convert '{value}' to boolean for argument '{key}'. Keeping default.")
                        else:
                            setattr(args, key, bool(value))
                    else:
                        try:
                            setattr(args, key, arg_type(value))
                        except ValueError:
                            print(f"Could not convert '{value}' to type {arg_type.__name__} for argument '{key}'. Keeping default.")
                else:
                    print(f"Config file contains unknown argument: '{key}'. Ignoring.")
        except Exception as e:
            print(f"Could not load config file '{args.config_path}': {e}. Using command-line arguments.")

    print(f"Using config: {args}")
    return args

# =========================================================================
# ======================== 新增的数据加载辅助函数 ==========================
# =========================================================================
def initialize_dataloaders_from_yaml(config_path, args):
    """
    从YAML配置文件中读取设置，并初始化训练和验证的数据加载器。
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"数据集配置文件未找到: {config_path}。请确保它与主脚本在同一目录下。")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if 'paired_dataset' not in config:
        raise ValueError("YAML配置文件中缺少 'paired_dataset' 顶级键。")
        
    config_dir = os.path.dirname(os.path.abspath(config_path))
    data_config = config['paired_dataset']
    dataloaders = {}
    
    for split in ['train', 'validate']:
        if split in data_config:
            dataset_config = data_config[split]
            logger.info(f"--- 正在准备 '{split}' 数据加载器 ---")
            
            # 动态导入数据集类，例如 'data_loader.PairedImageDataset'
            module_path, class_name = dataset_config['target'].rsplit('.', 1)
            try:
                module = importlib.import_module(module_path)
                DatasetClass = getattr(module, class_name)
            except (ImportError, AttributeError) as e:
                logger.error(f"错误：无法从 '{dataset_config['target']}' 导入类。请检查 target 路径是否正确。")
                raise e

            # 准备传递给数据集构造函数的参数
            params = dataset_config.get('params', {})
            # 从args中获取resolution，并允许YAML中的设置覆盖它
            params['resolution'] = int(params.get('resolution', args.resolution)) 
            params['config_dir'] = config_dir
            params['split'] = split # 显式传入split
            
            # 实例化数据集
            dataset = DatasetClass(**params)
            
            if len(dataset) == 0:
                logger.warning(f"!!! '{split}' 数据集为空，数据加载器将不会被创建。请检查数据路径和文件命名。")
                dataloaders[split] = None
                continue

            # 创建数据加载器
            # 由于图像被调整到固定大小，不再需要 BucketBatchSampler
            is_train = split == 'train'
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=args.train_batch_size,
                shuffle=is_train,
                num_workers=10 # 可根据需要调整
            )
            dataloaders[split] = dataloader
            logger.info(f"--- '{split}' 数据加载器准备就绪, 样本数: {len(dataset)} ---")

    return dataloaders.get('train'), dataloaders.get('validate')
# =========================================================================
# =========================================================================

@torch.no_grad()
def vae_encode(vae, image):
    """
    根据用户提供的源代码，处理单个图像的VAE编码。
    """
    # create tensor latent
    pixel_values = []
    pixel_values.append(image)
    # 将单个图像放入列表，然后堆叠成一个批次大小为1的张量
    pixel_values = torch.stack(pixel_values).to(vae.device, dtype=vae.dtype)
    
    with torch.no_grad():
        # VAE编码这个大小为1的批次
        # .sample() 从分布中采样
        latent = vae.encode(pixel_values).latent_dist.sample()
        # .squeeze(0) 移除批次维度，得到 (channels, height, width)
        latent = latent.squeeze(0)
        del pixel_values
    # 注意：原始函数返回到CPU，但我们在循环内使用，所以暂时保留在GPU上
    # latent_dict = {
    #     'latent': latent.cpu()
    # }
    return latent

def main(args):
    
    # --- 原有的变量设置保持不变 ---
    use_8bit_adam = True
    adam_beta1 = 0.9
    adam_beta2 = 0.99
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-08
    dataloader_num_workers = 0
    max_train_steps = None
    max_grad_norm = 1.0
    revision = None
    variant = None
    prodigy_decouple = True
    prodigy_beta3 = None
    prodigy_use_bias_correction = True
    prodigy_safeguard_warmup = True
    prodigy_d_coef = 2
    lr_power = 1
    val_seed = 42
    
    # --- 定义训练布局和图像键名 ---
    # 这个结构现在变得更简单，因为我们总是处理 lq/hq 对
    image_1 = "hq_image" # 高质量图像，将被加噪
    image_2 = "lq_image" # 低质量图像，作为参考
    
    transformer_subfolder = "transformer"

    # --- 训练布局配置 ---
    # 这个配置定义了在 train_process 中如何处理 lq/hq latent
    training_layout_configs = {
        image_1: { # hq_image
            "target": image_1,
            "noised": True, # 这个latent会被加噪
        },
        image_2: { # lq_image
            "target": image_2,
            "noised": False, # 这个latent作为参考，不加噪
        },
    }

    lr_num_cycles = args.cosine_restarts
    resolution = int(args.resolution)
    
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    if not os.path.exists(args.logging_dir): os.makedirs(args.logging_dir)
    
    # --- Accelerator 初始化 ---
    logging_dir = "logs"
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

        
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    run_name = f"{args.save_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    


    # 仅在主进程中设置文件日志，这是最佳实践
    log_level = "INFO"
    logger = get_logger(__name__, log_level=log_level)
    if accelerator.is_main_process:
        # 确保输出目录存在
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
        
        log_file_path = os.path.join(args.output_dir, "training_run.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level.upper())
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        
        # 将文件处理器添加到根 logger
        logging.getLogger().addHandler(file_handler)
        
        # 现在可以安全地使用 logger 了
        logger.info(f"文件日志已配置。所有详细日志将被保存到: {log_file_path}")


    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif accelerator.mixed_precision == "fp8":
        weight_dtype = torch.float8_e4m3fn
    
    # --- 加载 Scheduler ---
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler"
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # --- 加载 VAE (必须保留，因为现在需要在循环中编码图像) ---
    vae_path = args.vae_path if args.vae_path else args.pretrained_model_name_or_path
    vae_subfolder = "vae" if not args.vae_path else None
    logger.info("加载 VAE...")
    vae = AutoencoderKL.from_pretrained(
        vae_path,
        subfolder=vae_subfolder,
    )
    # 冻结VAE参数，因为它不参与训练
    vae.requires_grad_(False)
    # 将VAE移至GPU并设置为评估模式
    vae.to(accelerator.device, dtype=torch.float32) # VAE通常在float32下表现更好
    vae.eval()

    # =========================================================================
    # ======================= 新：一次性计算文本嵌入 ==========================
    # =========================================================================
    logger.info("加载文本编码器以计算固定Prompt的嵌入...")
    # 1. 加载 Tokenizer 和 Text Encoder
    tokenizer_one = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_two = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_cls_one = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    text_encoder_one, text_encoder_two = load_text_encoders(text_encoder_cls_one, text_encoder_cls_two)

    # 2. 移动到设备并设置数据类型
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.eval()
    text_encoder_two.eval()

    # 3. 计算固定Prompt的嵌入
    logger.info(f"正在为固定Prompt计算文本嵌入: '{FIXED_PROMPT}'")
    with torch.no_grad():
        fixed_prompt_embeds, fixed_pooled_prompt_embeds, txt_attention_masks = compute_text_embeddings(
            text_encoders=[text_encoder_one, text_encoder_two],
            tokenizers=[tokenizer_one, tokenizer_two],
            prompt=FIXED_PROMPT,
            device=accelerator.device
        )
        fixed_prompt_embeds = fixed_prompt_embeds.squeeze(0)
        fixed_pooled_prompt_embeds = fixed_pooled_prompt_embeds.squeeze(0)
        txt_attention_mask = txt_attention_masks.squeeze(0)
        
    
    # 4. **关键步骤**: 卸载文本编码器和分词器以释放显存
    logger.info("文本嵌入计算完成，正在卸载文本编码器以节省VRAM...")
    del text_encoder_one, text_encoder_two, tokenizer_one, tokenizer_two
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("文本编码器已卸载。")
    # =========================================================================
    # =========================================================================

    # =========================================================================
    # ===================== 新：从YAML加载数据集 ==============================
    # =========================================================================
    logger.info("正在从 config.yaml 初始化数据加载器...")
    train_dataloader, val_dataloader = initialize_dataloaders_from_yaml(CONFIG_YAML_PATH, args)
    if train_dataloader is None:
        logger.error("训练数据加载失败，无法继续。请检查 config.yaml 和数据路径。")
        return
    # =========================================================================
    # =========================================================================
    
    flush()
    
    # --- Transformer 模型加载和 LoRA 配置 (与原脚本相同) ---
    offload_device = accelerator.device
    if not (args.model_path is None or args.model_path == ""):
        transformer = MaskedFluxTransformer2DModel.from_single_file(args.model_path, torch_dtype=weight_dtype).to(offload_device)
    else:
        if args.pretrained_model_name_or_path == "black-forest-labs/FLUX.1-dev":
            transformer = MaskedFluxTransformer2DModel.from_pretrained(
                args.pretrained_model_name_or_path, subfolder=transformer_subfolder, torch_dtype=weight_dtype
            ).to(offload_device)
        else:
            transformer_folder = os.path.join(args.pretrained_model_name_or_path, transformer_subfolder)
            transformer = MaskedFluxTransformer2DModel.from_pretrained(
                transformer_folder, variant=None, torch_dtype=weight_dtype
            ).to(offload_device)
    flush()

    if "quantization_config" in transformer.config:
        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    else:
        transformer = transformer.to(offload_device, dtype=weight_dtype)
        transformer.requires_grad_(False)
    
    is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0
    if is_swapping_blocks:
        logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
        transformer.enable_block_swap(args.blocks_to_swap, accelerator.device)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
        
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        # 默认的 LoRA 目标模块
        target_modules = [ "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0", "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out", "ff.net.0.proj", "ff.net.2", "ff_context.net.0.proj", "ff_context.net.2"]

    transformer_lora_config = LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian", target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    layer_names = []
    freezed_layers = []
    if args.freeze_transformer_layers is not None and args.freeze_transformer_layers != '':
        splited_layers = args.freeze_transformer_layers.split()
        for layer in splited_layers:
            layer_name = int(layer.strip())
            freezed_layers.append(layer_name)
    # Freeze the layers
    for name, param in transformer.named_parameters():
        layer_names.append(name)
        if "transformer" in name:
            if '_orig_mod.' in name:
                name = name.replace('_orig_mod.', '')
            name_split = name.split(".")
            layer_order = name_split[1]
            if int(layer_order) in freezed_layers:
                param.requires_grad = False
    
    # --- 保存/加载钩子和优化器设置 (大部分与原脚本相同) ---
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        # ... (此处省略 save_model_hook 的代码，与原文件完全相同) ...
        if accelerator.is_main_process:
            transformer_lora_layers_to_save = None
            for model in models:
                if isinstance(model, type(unwrap_model(transformer))):
                    transformer_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(model))
                else:
                    raise ValueError(f"unexpected save model: {model.__class__}")
                weights.pop()

            FluxKontextPipeline.save_lora_weights(
                output_dir,
                transformer_lora_layers=transformer_lora_layers_to_save
            )
            
            last_part = os.path.basename(os.path.normpath(output_dir))
            file_path = f"{output_dir}/{last_part}.safetensors"
            ori_file = f"{output_dir}/pytorch_lora_weights.safetensors"
            if os.path.exists(ori_file):
                shutil.copy(ori_file, file_path)
            
            if args.config_path:
                shutil.copy(args.config_path, output_dir)
            
            # 保存数据加载配置
            if os.path.exists(CONFIG_YAML_PATH):
                shutil.copy(CONFIG_YAML_PATH, os.path.join(output_dir, "data_config.yaml"))

    def load_model_hook(models, input_dir):
        transformer_ = None
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(unwrap_model(transformer))):
                transformer_ = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        lora_state_dict = FluxKontextPipeline.lora_state_dict(input_dir)
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
        transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
        incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
        if incompatible_keys is not None:
            # check only for unexpected keys
            unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
            if unexpected_keys:
                logger.warning(
                    f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                    f" {unexpected_keys}. "
                )
    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize = [{"params": transformer_lora_parameters, "lr": args.learning_rate}]
    
      # Optimizer creation
    if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
        logger.warning(
            f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
            "Defaulting to adamW"
        )
        args.optimizer = "adamw"

    if use_8bit_adam and not args.optimizer.lower() == "adamw":
        logger.warning(
            f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
            f"set to {args.optimizer.lower()}"
        )

    if args.optimizer.lower() == "adamw":
        if use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )

            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW

        optimizer = optimizer_class(
            params_to_optimize,
            betas=(adam_beta1, adam_beta2),
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
        )

    if args.optimizer.lower() == "prodigy":
        try:
            import prodigyopt
        except ImportError:
            raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

        optimizer_class = prodigyopt.Prodigy

        if args.learning_rate <= 0.1:
            logger.warning(
                "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
            )

        optimizer = optimizer_class(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(adam_beta1, adam_beta2),
            beta3=prodigy_beta3,
            d_coef=prodigy_d_coef,
            weight_decay=adam_weight_decay,
            eps=adam_epsilon,
            decouple=prodigy_decouple,
            use_bias_correction=prodigy_use_bias_correction,
            safeguard_warmup=prodigy_safeguard_warmup,
        )
        
        
        
        
        
    # Scheduler and math around the number of training steps.
    override_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if max_train_steps is None:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        override_max_train_steps = True

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if override_max_train_steps:
        max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)


    # vae config from vae config file
    vae_config_shift_factor = 0.1159
    vae_config_scaling_factor = 0.3611


    print("  Num examples = ", len(train_dataloader.dataset))
    print("  Num Epochs = ", args.num_train_epochs)
    print("  num_update_steps_per_epoch = ", num_update_steps_per_epoch)
    print("  max_train_steps = ", max_train_steps)

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=max_train_steps * accelerator.num_processes,
        num_cycles=lr_num_cycles,
        power=lr_power,
    )
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        optimizer, train_dataloader, lr_scheduler
    )
    
    # load transformer to cpu
    transformer.to("cuda")
    flush()
    
    transformer = accelerator.prepare(transformer, device_placement=[not is_swapping_blocks])
    
    
    
    

    # VAE 不需要被 prepare，因为它不训练，但要确保它在正确的设备上
    vae.to(accelerator.device)

    # --- Tracker 初始化 (与原脚本相同) ---
    if accelerator.is_main_process:
        try:
            accelerator.init_trackers("flux-lora", config=vars(args))
        except Exception as e:
            logger.warning(f"Trackers not initialized: {e}")

    # --- 训练信息打印 (与原脚本相同) ---
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    resume_step = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint and args.resume_from_checkpoint != "":
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the mos recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith(args.save_name)]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[-1])

            initial_global_step = global_step
            resume_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch

            transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
            # Optimization parameters
            transformer_lora_parameters_with_lr = {"params": transformer_lora_parameters, "lr": args.learning_rate}
            params_to_optimize = [transformer_lora_parameters_with_lr]
            
            # Optimizer creation
            if not (args.optimizer.lower() == "prodigy" or args.optimizer.lower() == "adamw"):
                logger.warning(
                    f"Unsupported choice of optimizer: {args.optimizer}.Supported optimizers include [adamW, prodigy]."
                    "Defaulting to adamW"
                )
                args.optimizer = "adamw"

            if use_8bit_adam and not args.optimizer.lower() == "adamw":
                logger.warning(
                    f"use_8bit_adam is ignored when optimizer is not set to 'AdamW'. Optimizer was "
                    f"set to {args.optimizer.lower()}"
                )

            if args.optimizer.lower() == "adamw":
                if use_8bit_adam:
                    try:
                        import bitsandbytes as bnb
                    except ImportError:
                        raise ImportError(
                            "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                        )

                    optimizer_class = bnb.optim.AdamW8bit
                else:
                    optimizer_class = torch.optim.AdamW

                optimizer = optimizer_class(
                    params_to_optimize,
                    betas=(adam_beta1, adam_beta2),
                    weight_decay=adam_weight_decay,
                    eps=adam_epsilon,
                )

            if args.optimizer.lower() == "prodigy":
                try:
                    import prodigyopt
                except ImportError:
                    raise ImportError("To use Prodigy, please install the prodigyopt library: `pip install prodigyopt`")

                optimizer_class = prodigyopt.Prodigy

                if args.learning_rate <= 0.1:
                    logger.warning(
                        "Learning rate is too low. When using prodigy, it's generally better to set learning rate around 1.0"
                    )

                optimizer = optimizer_class(
                    params_to_optimize,
                    lr=args.learning_rate,
                    betas=(adam_beta1, adam_beta2),
                    beta3=prodigy_beta3,
                    d_coef=prodigy_d_coef,
                    weight_decay=adam_weight_decay,
                    eps=adam_epsilon,
                    decouple=prodigy_decouple,
                    use_bias_correction=prodigy_use_bias_correction,
                    safeguard_warmup=prodigy_safeguard_warmup,
                )
            
            lr_scheduler = get_scheduler(
                args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
                num_training_steps=max_train_steps * accelerator.num_processes,
                num_cycles=lr_num_cycles,
                power=lr_power,
            )
            
            optimizer, lr_scheduler = accelerator.prepare(optimizer, lr_scheduler)
    else:
        initial_global_step = 0
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )



    if accelerator.unwrap_model(transformer).config.guidance_embeds:
        handle_guidance = True
    else:
        handle_guidance = False
        
    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma
        
    # =========================================================================
    # ======================== 重构后的 train_process =========================
    # =========================================================================
    def train_process(
            batch,
            vae, # 新增 VAE 模型作为参数
            fixed_prompt_embeds, # 新增固定的文本嵌入
            fixed_pooled_prompt_embeds, # 新增固定的池化嵌入
        ):
        
        accelerator.unwrap_model(transformer).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(transformer).prepare_block_swap_before_forward()
        flush()

        # 1. 从批次中获取图像并计算时间步
        hq_images = batch['hq_image'].to(dtype=weight_dtype)
        lq_images = batch['lq_image'].to(dtype=weight_dtype)
        batch_size = hq_images.shape[0]
        
        vae_config_block_out_channels = [128,256,512,512]
   

        
        u = compute_density_for_timestep_sampling(
            weighting_scheme=args.weighting_scheme, batch_size=batch_size, 
            logit_mean=args.logit_mean, logit_std=args.logit_std, mode_scale=args.mode_scale
        )
        indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
        timesteps = noise_scheduler_copy.timesteps[indices].to(device=accelerator.device)

      # =========================================================================
        # =================== 精确复现源代码的VAE编码和处理逻辑 ===================
        # =========================================================================
        
        # 1. 遍历批次，使用用户提供的 vae_encode 函数处理单张图片
        hq_latents_list = [vae_encode(vae, hq_images[i]) for i in range(batch_size)]
        lq_latents_list = [vae_encode(vae, lq_images[i]) for i in range(batch_size)]

        # 2. 将 latent 列表堆叠成批次张量
        hq_latents_batch = torch.stack(hq_latents_list)
        lq_latents_batch = torch.stack(lq_latents_list)

        # 3. 严格按照源代码的公式进行 shift 和 scale
        hq_latents = (hq_latents_batch - vae_config_shift_factor) * vae_config_scaling_factor
        lq_latents = (lq_latents_batch - vae_config_shift_factor) * vae_config_scaling_factor
        
        # 4. 准备latents列表以匹配原始逻辑
        noised_latent_list = [hq_latents]
        target_list = [hq_latents]
        latent_list = [lq_latents]
        # =========================================================================
        # =========================================================================

        noised_latents = torch.cat(noised_latent_list, dim=0)
        noise = torch.randn_like(noised_latents) + args.noise_offset * torch.randn(
            noised_latents.shape[0], noised_latents.shape[1], 1, 1).to(accelerator.device)
        
        sigmas = get_sigmas(timesteps, n_dim=noised_latents.ndim, dtype=noised_latents.dtype)
        noisy_model_input = (1.0 - sigmas) * noised_latents + sigmas * noise
        latents = noisy_model_input

        # 4. 打包latents (与原始逻辑相同)
        packed_noisy_latents = FluxKontextPipeline._pack_latents(
            noisy_model_input, batch_size=latents.shape[0], num_channels_latents=latents.shape[1],
            height=latents.shape[2], width=latents.shape[3],
        )

        ref_latents = torch.cat(latent_list, dim=0)
        packed_ref_latents = FluxKontextPipeline._pack_latents(
            ref_latents, batch_size=ref_latents.shape[0], num_channels_latents=ref_latents.shape[1],
            height=ref_latents.shape[2], width=ref_latents.shape[3],
        )
        
        ref_image_ids = None
        packed_ref_latents = None
        # handle partial noised
        if len(latent_list) > 0:
            ref_latents = torch.cat(latent_list, dim=0)   
            # pack noisy latents
            packed_ref_latents = FluxKontextPipeline._pack_latents(
                ref_latents,
                batch_size=ref_latents.shape[0],
                num_channels_latents=ref_latents.shape[1],
                height=ref_latents.shape[2],
                width=ref_latents.shape[3],
            )
            ref_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                ref_latents.shape[0],
                ref_latents.shape[2] // 2,
                ref_latents.shape[3] // 2,
                accelerator.device,
                weight_dtype,
            )
            ref_image_ids[..., 0] = 1
        
        # cat factual_images as image guidance
        learning_target = torch.cat(target_list, dim=0)
        
        latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
            latents.shape[0],
            latents.shape[2] // 2,
            latents.shape[3] // 2,
            accelerator.device,
            weight_dtype,
        )
        
        if ref_image_ids is not None:
            latent_image_ids = torch.cat([latent_image_ids, ref_image_ids], dim=0)  # dim 0 is sequence dimension

        model_input = packed_noisy_latents
        # add ref to channel
        if packed_ref_latents is not None:
            model_input = torch.cat((packed_noisy_latents, packed_ref_latents), dim=1)
            
        if handle_guidance:
            guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        # 5. **核心修改**: 使用预先计算好的、固定的文本嵌入
        # 扩展嵌入以匹配当前批次大小
        prompt_embeds = fixed_prompt_embeds.expand(batch_size, -1, -1).to(device=accelerator.device, dtype=weight_dtype)
        pooled_prompt_embeds = fixed_pooled_prompt_embeds.expand(batch_size, -1).to(device=accelerator.device, dtype=weight_dtype)
        
        txt_attention_masks = None
        # 根据caption_dropout概率应用无条件训练
        if random.random() < args.caption_dropout:
            prompt_embeds = torch.zeros_like(prompt_embeds)
            pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=accelerator.device, dtype=weight_dtype)
        
        # 6. 前向传播和损失计算 (与原始逻辑几乎完全相同)
        with accelerator.autocast():
            model_pred = transformer(
                hidden_states=model_input,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                timestep=timesteps / 1000,
                img_ids=latent_image_ids,
                txt_ids=text_ids,
                guidance=guidance,
                return_dict=False
            )[0]
        
        model_pred = model_pred[:, : packed_noisy_latents.size(1)]
        
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        model_pred = FluxKontextPipeline._unpack_latents(
            model_pred, height=latents.shape[2] * vae_scale_factor, width=latents.shape[3] * vae_scale_factor,
            vae_scale_factor=vae_scale_factor,
        )

        learning_target = torch.cat(target_list, dim=0)
        target = noise - learning_target
        _,_,_,t_w = target.shape
        _,_,_,p_w = model_pred.shape
        # split model_pred based on target width
        if p_w != t_w:
            model_pred = model_pred[..., :t_w]
        
        weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
        
        # Compute regular loss.
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
            1,
        )
        
        loss = loss.mean()
        
        total_loss = loss
        
        return total_loss
    # =========================================================================
    # =========================================================================


    # --- 主训练循环 ---
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            with accelerator.accumulate(transformer):
                # 调用重构后的 train_process
                loss = train_process(
                    batch,
                    vae,
                    fixed_prompt_embeds,
                    fixed_pooled_prompt_embeds,
                )

                accelerator.backward(loss)
                step_loss = loss.detach().item()
                if accelerator.sync_gradients:
                    params_to_clip = transformer_lora_parameters
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)

                del loss
                flush()
                # ensure model in cuda
                transformer.to(accelerator.device)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Checks if the accelerator has performed an optimization step behind the scenes
                #post batch check for gradient updates
                # accelerator.wait_for_everyone()
                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                
                lr = lr_scheduler.get_last_lr()[0]
                lr_name = "lr"
                if args.optimizer == "prodigy":
                    if resume_step>0 and resume_step == global_step:
                        lr = 0
                    else:
                        lr = lr_scheduler.optimizers[-1].param_groups[0]["d"] * lr_scheduler.optimizers[-1].param_groups[0]["lr"]
                    lr_name = "lr/d*lr"
                logs = {"step_loss": step_loss, lr_name: lr}
                accelerator.log(logs, step=global_step)
                progress_bar.set_postfix(**logs)
                
                if global_step >= max_train_steps:
                    break
                # del step_loss
                flush()

                # --- 保存和验证逻辑 (结构与原脚本相同，但调用方式已更新) ---
                if global_step >= max_train_steps:
                    break
                
                args.save_model_steps = 2000
                # 按步数保存
                if args.save_model_steps > 0 and global_step % args.save_model_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(args.output_dir, f"{args.save_name}-step-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")
                        
                # ==============================================================================
                # --- 以下是为您补充的“按步数进行验证”的代码块 ---
                # ==============================================================================
                # 检查是否达到了预设的验证步数，并且验证数据加载器 (val_dataloader) 存在。
                if args.save_model_steps > 0 and global_step % args.save_model_steps == 0 and val_dataloader is not None:
                    # 记录日志，表示验证开始。
                    logger.info("***** Running Validation (by step) *****")
                    # 将模型切换到评估（evaluation）模式。这会关闭 Dropout 和 BatchNorm 等层。
                    transformer.eval()
                    # 初始化总验证损失。
                    total_val_loss = 0.0
                    
                    # 使用 torch.no_grad() 上下文管理器，在此代码块中不计算梯度，以节省显存和计算资源。
                    with torch.no_grad():
                        # 遍历验证数据加载器，并用 tqdm 显示进度条。
                        for val_batch in tqdm(val_dataloader, desc="Validating", disable=not accelerator.is_local_main_process):
                            # 验证时也调用同样的 train_process 函数来计算损失。
                            val_loss = train_process(
                                val_batch,
                                vae,
                                fixed_prompt_embeds,
                                fixed_pooled_prompt_embeds,
                            )
                            # 使用 accelerator.gather(...) 来同步所有分布式进程（GPU）上的损失值。
                            # 如果只有一个GPU，它会直接返回张量。
                            # repeat() 是为了让每个进程的张量形状一致，这是 gather 的要求。
                            gathered_loss = accelerator.gather(val_loss.repeat(args.train_batch_size))
                            # 累加所有GPU上的平均损失。
                            total_val_loss += torch.mean(gathered_loss).item()
    
                    # 计算所有批次的平均验证损失。
                    avg_val_loss = total_val_loss / len(val_dataloader)
                    # 记录详细的验证损失日志。
                    logger.info(f"Step {global_step}: Validation Loss = {avg_val_loss}")
                    # 将验证损失记录到 wandb/tensorboard 等追踪工具中。
                    accelerator.log({"val_loss_step": avg_val_loss}, step=global_step)
                    # !!! 非常重要：验证结束后，必须将模型切换回训练模式，以便继续训练 !!!
                    transformer.train()
                # ==============================================================================
                # --- 补充代码结束 ---
                # ==============================================================================


        # 按 epoch 保存和验证
        if (epoch + 1) % args.save_model_epochs == 0 or (epoch + 1) == args.num_train_epochs:
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f"{args.save_name}-epoch-{epoch+1}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

        # 运行验证
        if (epoch + 1) % args.validation_epochs == 0 and val_dataloader is not None:
            logger.info("***** Running Validation *****")
            transformer.eval()
            total_val_loss = 0.0
            
            with torch.no_grad():
                for val_batch in tqdm(val_dataloader, desc="Validating", disable=not accelerator.is_local_main_process):
                    # 验证时也调用同样的 train_process 函数来计算损失
                    val_loss = train_process(
                        val_batch,
                        vae,
                        fixed_prompt_embeds,
                        fixed_pooled_prompt_embeds,
                    )
                    # 收集所有GPU上的损失
                    gathered_loss = accelerator.gather(val_loss.repeat(args.train_batch_size))
                    total_val_loss += torch.mean(gathered_loss).item()

            avg_val_loss = total_val_loss / len(val_dataloader)
            logger.info(f"Epoch {epoch+1}: Validation Loss = {avg_val_loss}")
            accelerator.log({"val_loss": avg_val_loss}, step=global_step)
            transformer.train() # 验证后切换回训练模式
        
        if global_step >= max_train_steps:
            break
            
        gc.collect()
        torch.cuda.empty_cache()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # 保存最终模型
        transformer = unwrap_model(transformer)
        final_save_path = os.path.join(args.output_dir, f"{args.save_name}-final")
        accelerator.save_state(final_save_path)
        logger.info(f"Final model state saved to {final_save_path}")

    accelerator.end_training()
    print("训练完成。模型保存在: ")
    print(args.output_dir)


if __name__ == "__main__":
    args = parse_args()
    main(args)