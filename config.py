import sys, os
from pathlib import Path
import torch

# 推理用的指定模型
sovits_path = ""
gpt_path = ""
is_half_str = os.environ.get("is_half", "True")
is_half = True if is_half_str.lower() == 'true' else False
BASE_DIR = Path(__file__).resolve().parent

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# 人声提取激进程度 0-20，默认10
agg = 5
input_language = "ja"
output_language = "zh"
# 重试次数
retry_times = 5
TEMP_PATH = os.path.join(BASE_DIR, "TEMP")


cnhubert_path = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
bert_path = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
pretrained_sovits_path = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_path = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
uvr5_weights_path = "tools/uvr5/uvr5_weights"
asr_models_path = "tools/asr/models"
resemble_enhance_model_path = os.path.join(BASE_DIR, 'tools/resemble_enhance/model_repo/enhancer_stage2/')
resemble_enhance_cmd = f"/root/miniconda3/envs/resemble-enhance/bin/resemble-enhance --run_dir {resemble_enhance_model_path}"

exp_root = "logs"
# 字幕用于处理的目标style，默认会通过筛选出场率最高的style作为目标处理字幕
sub_style = ""

# 是否合成mkv文件
combined_mkv = 1

python_exec = sys.executable or "python"
if torch.cuda.is_available():
    infer_device = "cuda"
else:
    infer_device = "cpu"

if infer_device == "cuda":
    gpu_name = torch.cuda.get_device_name(0)
    if (
            ("16" in gpu_name and "V100" not in gpu_name.upper())
            or "P40" in gpu_name.upper()
            or "P10" in gpu_name.upper()
            or "1060" in gpu_name
            or "1070" in gpu_name
            or "1080" in gpu_name
    ):
        is_half = False

if infer_device == "cpu": is_half = False


class Config:
    def __init__(self):
        self.sovits_path = sovits_path
        self.gpt_path = gpt_path
        self.is_half = is_half

        self.cnhubert_path = cnhubert_path
        self.bert_path = bert_path
        self.pretrained_sovits_path = pretrained_sovits_path
        self.pretrained_gpt_path = pretrained_gpt_path

        self.exp_root = exp_root
        self.python_exec = python_exec
        self.infer_device = infer_device
        self.resemble_enhance_cmd = resemble_enhance_cmd
