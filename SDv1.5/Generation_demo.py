import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import argparse
import os
import numpy as np
import torch
#torch.cuda.set_device(0)
from pathlib import Path
# from PIL import Image
from IPython.display import display
from config import RunConfig
from run import run
from notebooks.prompts_dict import sketchy_prompts, common_prompts

#remove the last generation result
# if os.path.exists('./neurips_demo'):
#     import shutil
#     shutil.rmtree('./neurips_demo')
    
prompts=common_prompts

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=float, default=15.0)
parser.add_argument('--swap_guidance_scale', type=float, default=15.0)
parser.add_argument('--interpolation', type=float, default=0.1)

args = parser.parse_args()


skip_steps = 10
CFG = args.cfg
swap_guidance_scale = args.swap_guidance_scale
for j in [1,2,3,4,5]:
  style_image_path = f"./style/style{j}/"
  image_path_list = sorted(os.listdir(style_image_path))*3
  image_path_list = image_path_list[0:len(prompts)]
  seed = 42 
  sparse_weight = 0
  interpolation = args.interpolation
  n_images = len(image_path_list)
  for i, image_path in enumerate(image_path_list):
      print(prompts[i+1])
      full_image_path = os.path.join(style_image_path, image_path_list[i])
      full_image_path1 = os.path.join(style_image_path, image_path_list[(i+1) % n_images])
      full_image_path2 = os.path.join(style_image_path, image_path_list[(i+2) % n_images])
      
      domain_name = prompts[i+1]
      config = RunConfig(
          skip_steps=skip_steps,
          app_image_path=Path(full_image_path),
          struct_image_path=Path(full_image_path1),
          style_image_path=Path(full_image_path2),
          output_path=Path(f'neurips_demo/CFG{CFG}_sfg{swap_guidance_scale}_skip{skip_steps}/style{j}_{sparse_weight}_int_{interpolation}'),
          #output_path=Path(f'kv_swap/style{j}/'),
          domain_name=domain_name,
          seed=seed,
          swap_guidance_scale=swap_guidance_scale,
          CFG=CFG,
          mix_style=True,
          sparse_weight=sparse_weight,
          load_latents=False,
          interpolation=interpolation,
          resize=False,
          alpha=0.5,
          reimu=0.3
      )
      images = run(cfg=config)
      torch.cuda.empty_cache()

