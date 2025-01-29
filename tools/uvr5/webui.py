import os
import logging
from tools.i18n.i18n import I18nAuto
from tools.my_utils import clean_path
i18n = I18nAuto()

logger = logging.getLogger(__name__)
import librosa,ffmpeg
import soundfile as sf
import torch
import sys
from mdxnet import MDXNetDereverb
from vr import AudioPre, AudioPreDeEcho
from bsroformer import BsRoformer_Loader


weight_uvr5_root = "tools/uvr5/uvr5_weights"
device="cuda"
is_half=True
agg = 10 # 1 - 20




try:
    
    pre_fun = AudioPre(
        agg=int(10),
        model_path=os.path.join(weight_uvr5_root, "HP5_only_main_vocal.pth"),
        device=device,
        is_half=is_half,
    )
    
    pre_fun._path_audio_(
        music_file="/home/data/liujiaqi/Ai/GPT-SoVITS/TEMP/input.mp3", 
        ins_root="/home/data/liujiaqi/Ai/GPT-SoVITS/TEMP/uvr5/instrument",
        vocal_root="/home/data/liujiaqi/Ai/GPT-SoVITS/TEMP/uvr5/vocal", 
        format="wav", 
        is_hp3=False
    )
except Exception as err:
    print(err)
finally:
    try:
        del pre_fun.model
        del pre_fun
    except Exception as err:
        print(err)
    print("clean_empty_cache")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



            
                   
# with gr.Blocks(title="UVR5 WebUI") as app:
#     gr.Markdown(
#         value=
#             i18n("本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责. <br>如不认可该条款, 则不能使用或引用软件包内任何代码和文件. 详见根目录<b>LICENSE</b>.")
#     )
#     with gr.Group():
#         gr.Markdown(html_center(i18n("伴奏人声分离&去混响&去回声"),'h2'))
#         with gr.Group():
#                 gr.Markdown(
#                     value=html_left(i18n("人声伴奏分离批量处理， 使用UVR5模型。") + "<br>" + \
#                         i18n("合格的文件夹路径格式举例： E:\\codes\\py39\\vits_vc_gpu\\白鹭霜华测试样例(去文件管理器地址栏拷就行了)。")+ "<br>" + \
#                         i18n("模型分为三类：") + "<br>" + \
#                         i18n("1、保留人声：不带和声的音频选这个，对主人声保留比HP5更好。内置HP2和HP3两个模型，HP3可能轻微漏伴奏但对主人声保留比HP2稍微好一丁点；") + "<br>" + \
#                         i18n("2、仅保留主人声：带和声的音频选这个，对主人声可能有削弱。内置HP5一个模型；") + "<br>" + \
#                         i18n("3、去混响、去延迟模型（by FoxJoy）：") + "<br>  " + \
#                         i18n("(1)MDX-Net(onnx_dereverb):对于双通道混响是最好的选择，不能去除单通道混响；") + "<br>&emsp;" + \
#                         i18n("(234)DeEcho:去除延迟效果。Aggressive比Normal去除得更彻底，DeReverb额外去除混响，可去除单声道混响，但是对高频重的板式混响去不干净。") + "<br>" + \
#                         i18n("去混响/去延迟，附：") + "<br>" + \
#                         i18n("1、DeEcho-DeReverb模型的耗时是另外2个DeEcho模型的接近2倍；") + "<br>" + \
#                         i18n("2、MDX-Net-Dereverb模型挺慢的；") + "<br>" + \
#                         i18n("3、个人推荐的最干净的配置是先MDX-Net再DeEcho-Aggressive。"),'h4')
#                 )
#                 with gr.Row():
#                     with gr.Column():
#                         model_choose = gr.Dropdown(label=i18n("模型"), choices=uvr5_names)
#                         dir_wav_input = gr.Textbox(
#                             label=i18n("输入待处理音频文件夹路径"),
#                             placeholder="C:\\Users\\Desktop\\todo-songs",
#                         )
#                         wav_inputs = gr.File(
#                             file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
#                         )
#                     with gr.Column():
#                         agg = gr.Slider(
#                             minimum=0,
#                             maximum=20,
#                             step=1,
#                             label=i18n("人声提取激进程度"),
#                             value=10,
#                             interactive=True,
#                             visible=False,  # 先不开放调整
#                         )
#                         opt_vocal_root = gr.Textbox(
#                             label=i18n("指定输出主人声文件夹"), value="TEMP/uvr5_opt"
#                         )
#                         opt_ins_root = gr.Textbox(
#                             label=i18n("指定输出非主人声文件夹"), value="TEMP/uvr5"
#                         )
#                         format0 = gr.Radio(
#                             label=i18n("导出文件格式"),
#                             choices=["wav", "flac", "mp3", "m4a"],
#                             value="flac",
#                             interactive=True,
#                         )
#                         with gr.Column():
#                             with gr.Row():
#                                 but2 = gr.Button(i18n("转换"), variant="primary")
#                             with gr.Row():
#                                 vc_output4 = gr.Textbox(label=i18n("输出信息"),lines=3)
#                     but2.click(
#                         uvr,
#                         [
#                             model_choose,
#                             dir_wav_input,
#                             opt_vocal_root,
#                             wav_inputs,
#                             opt_ins_root,
#                             agg,
#     format0,)
                        
