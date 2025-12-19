from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from appearance_transfer_model import AppearanceTransferModel
from config import RunConfig
from utils import image_utils
from utils.ddpm_inversion import invert


def load_latents_or_invert_images(model: AppearanceTransferModel, cfg: RunConfig):
    # TODO: add support for style image
    print(cfg.app_latent_save_path)
    if cfg.load_latents and cfg.app_latent_save_path.exists() and cfg.struct_latent_save_path.exists():
        print("Loading existing latents...")
        latents_app, latents_struct, latent_style = load_latents(
            cfg.app_latent_save_path, cfg.struct_latent_save_path)
        noise_app, noise_struct, noise_style = load_noise(
            cfg.app_latent_save_path, cfg.struct_latent_save_path)
        print("Done.")
    else:
        print("Inverting images...")
        app_image, struct_image, style_image = image_utils.load_images(
            cfg=cfg, save_path=cfg.output_path)
        model.enable_edit = False  # Deactivate the cross-image attention layers
        latents_app, latents_struct, latent_style, noise_app, noise_struct, noise_style = invert_images(app_image=app_image,
                                                                                                        struct_image=struct_image,
                                                                                                        style_image=style_image,
                                                                                                        sd_model=model.pipe,
                                                                                                        cfg=cfg) #type:ignore
        model.enable_edit = True
        print("Done.")
    return latents_app, latents_struct,latent_style, noise_app, noise_struct, noise_style


def load_latents(app_latent_save_path: Path, struct_latent_save_path: Path, style_latent_save_path: Optional[Path] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    latents_app = torch.load(app_latent_save_path)
    latents_struct = torch.load(struct_latent_save_path)
    latent_style = torch.load(style_latent_save_path)

    if type(latents_struct) == list:
        latents_app = [l.to("cuda") for l in latents_app]
        latents_struct = [l.to("cuda") for l in latents_struct]
        latent_style = [l.to("cuda") for l in latent_style]
    else:
        latents_app = latents_app.to("cuda")
        latents_struct = latents_struct.to("cuda")
        latent_style = latent_style.to("cuda")

    return latents_app, latents_struct, latent_style


def load_noise(app_latent_save_path: Path, struct_latent_save_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError("DDPM noise loading is not implemented yet.")
    latents_app = torch.load(app_latent_save_path.parent /
                             ('app_'+app_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_struct = torch.load(struct_latent_save_path.parent /
                                ('struct_'+struct_latent_save_path.stem + "_ddpm_noise.pt"))
    latents_app = latents_app.to("cuda")
    latents_struct = latents_struct.to("cuda")
    return latents_app, latents_struct


def invert_images(sd_model: AppearanceTransferModel, app_image: Image.Image, struct_image: Image.Image, style_image: Image.Image, cfg: RunConfig):
    input_app = torch.from_numpy(np.array(app_image)).float() / 127.5 - 1.0
    input_struct = torch.from_numpy(
        np.array(struct_image)).float() / 127.5 - 1.0
    input_style = torch.from_numpy(np.array(style_image)).float() / 127.5 - 1.0
    
    if cfg.resize:
        input_app = crop_and_resize(input_app)
        input_struct = crop_and_resize(input_struct)
        input_style = crop_and_resize(input_style)

    print("Inverting appearance image...")
    noise_app, latents_app = invert(x0=input_app.permute(2, 0, 1).unsqueeze(0).to('cuda'),
                                 pipe=sd_model,
                                 prompt_src=cfg.prompt_app,
                                 num_diffusion_steps=cfg.num_timesteps,
                                 cfg_scale_src=cfg.CFG)
    print("Inverting struct image...")
    noise_struct, latents_struct = invert(
        x0=input_struct.permute(2, 0, 1).unsqueeze(0).to('cuda'),
        pipe=sd_model,
        prompt_src=cfg.prompt_struct,
        num_diffusion_steps=cfg.num_timesteps,
        cfg_scale_src=cfg.CFG
    )
    print("inverting style image...")
    noise_style, latent_style = invert(
        x0=input_style.permute(2, 0, 1).unsqueeze(0).to('cuda'),
        pipe=sd_model,
        prompt_src=cfg.prompt_style,
        num_diffusion_steps=cfg.num_timesteps,
        cfg_scale_src=cfg.CFG
    )
    
    # Save the inverted latents and noises
    torch.save(latents_app, cfg.latents_path /
               f"app_{cfg.app_image_path.stem}.pt")
    torch.save(latents_struct, cfg.latents_path /
               f"struct_{cfg.struct_image_path.stem}.pt")
    torch.save(noise_app, cfg.latents_path /
               f"app_{cfg.app_image_path.stem}_ddpm_noise.pt")
    torch.save(noise_struct, cfg.latents_path /
               f"struct_{cfg.struct_image_path.stem}_ddpm_noise.pt")
    torch.save(latent_style, cfg.latents_path /
               f"style_{cfg.style_image_path.stem}.pt")
    torch.save(noise_style, cfg.latents_path /
               f"style_{cfg.style_image_path.stem}_ddpm_noise.pt")
    
    return latents_app, latents_struct, latent_style, noise_app, noise_struct, noise_style


def get_init_latents_and_noises(model: AppearanceTransferModel, cfg: RunConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    # If we stored all the latents along the diffusion process, select the desired one based on the skip_steps
    # TODO: add cat for style latent
    if model.latents_struct.dim() == 4 and model.latents_app.dim() == 4 and model.latent_style.dim() == 4 and model.latents_app.shape[0] > 1:
        model.latents_struct = model.latents_struct[cfg.skip_steps]
        model.latents_app = model.latents_app[cfg.skip_steps]
        model.latent_style = model.latent_style[cfg.skip_steps]
        
        if cfg.skip_steps == -1:
            print("Initialize the latents.")
            # torch.randn_like(model.latents_struct)
            z_T = torch.load("source_noise.pt")
            # torch.save(z_T, "source_noise.pt")
        else:
            z_T = model.latents_struct

    init_latents = torch.stack([z_T, model.latents_app, model.latents_struct, model.latent_style])
    init_zs = [model.zs_struct[cfg.skip_steps:],
               model.zs_app[cfg.skip_steps:], 
               model.zs_struct[cfg.skip_steps:],
               model.zs_style[cfg.skip_steps:]]
    return init_latents, init_zs


def crop_and_resize(img, target_size=(512, 512), threshold=0.85):
    img_np = np.array(img)
    mask = img_np.mean(axis=2, keepdims=True) < threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        print("all white")
        return img
    y0, x0, _ = coords.min(axis=0)
    y1, x1, _ = coords.max(axis=0) + 1
    if np.abs(x0-x1)*np.abs(y0-y1) > (256*256):
        return img
    cropped_img = img[y0:y1, x0:x1, :]  # img.crop((x0, y0, x1, y1))
    cropped_img = cropped_img.permute(2, 0, 1)
    print(cropped_img.shape)
    pad_length = y1-y0 if y1-y0 > x1-x0 else x1-x0
    pad_length = int(pad_length/2.5)

    rotat = transforms.Compose([transforms.RandomRotation(degrees=(-30, 30), fill=1),
                                transforms.RandomHorizontalFlip(0.5),
                                ])
    if np.abs(x0-x1)*np.abs(y0-y1) < (128*128):
        cropped_img = torch.cat([cropped_img, rotat(cropped_img)], dim=1)
        cropped_img = torch.cat([cropped_img, rotat(cropped_img)], dim=2)
    else:
        if np.abs(x0-x1) < np.abs(y0-y1):
            cropped_img = torch.cat([cropped_img, rotat(cropped_img)], dim=2)
        else:
            cropped_img = torch.cat([cropped_img, rotat(cropped_img)], dim=1)

    transform = transforms.Compose([
        transforms.Pad(pad_length, fill=1, padding_mode='constant'),
        transforms.Resize(target_size, transforms.InterpolationMode.BICUBIC)
    ])
    resized_img = transform(cropped_img)
    print('image is croped and resized')
    return resized_img.permute(1, 2, 0)
