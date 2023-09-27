# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import logging
import os
import shutil

import torch
from cuda import cudart
from typing import Union

from diffusion_models import CLIP, VAE, CLIPWithProj, UNet, UNetXL

import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import CudaSession

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------------------------------
# Utilities for CUDA EP
# -----------------------------------------------------------------------------------------------------


class Engine(CudaSession):
    def __init__(self, engine_path, provider: str, device_id: int = 0, enable_cuda_graph=False):
        self.engine_path = engine_path
        self.provider = provider
        self.provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)

        device = torch.device("cuda", device_id)
        ort_session = ort.InferenceSession(
            self.engine_path,
            providers=[
                (provider, self.provider_options),
                "CPUExecutionProvider",
            ],
        )

        super().__init__(ort_session, device, enable_cuda_graph)


class Engines:
    def __init__(self, provider, onnx_opset: int = 14):
        self.provider = provider
        self.engines = {}
        self.onnx_opset = onnx_opset

    @staticmethod
    def get_onnx_path(onnx_dir, model_name):
        return os.path.join(onnx_dir, model_name + ".onnx")

    @staticmethod
    def get_engine_path(engine_dir, model_name, profile_id):
        return os.path.join(engine_dir, model_name + profile_id + ".onnx")

    def build(
        self,
        models,
        engine_dir: str,
        onnx_dir: str,
        force_engine_rebuild: bool = False,
        fp16: bool = True,
        device_id: int = 0,
        enable_cuda_graph: bool = False,
    ):
        profile_id = "_fp16" if fp16 else "_fp32"

        if force_engine_rebuild:
            if os.path.isdir(onnx_dir):
                logger.info("Remove existing directory %s since force_engine_rebuild is enabled", onnx_dir)
                shutil.rmtree(onnx_dir)
            if os.path.isdir(engine_dir):
                logger.info("Remove existing directory %s since force_engine_rebuild is enabled", engine_dir)
                shutil.rmtree(engine_dir)

        if not os.path.isdir(engine_dir):
            os.makedirs(engine_dir)

        if not os.path.isdir(onnx_dir):
            os.makedirs(onnx_dir)

        # Export models to ONNX
        for model_name, model_obj in models.items():
            onnx_path = Engines.get_onnx_path(onnx_dir, model_name)
            onnx_opt_path = Engines.get_engine_path(engine_dir, model_name, profile_id)
            if os.path.exists(onnx_opt_path):
                logger.info("Found cached optimized model: %s", onnx_opt_path)
            else:
                if os.path.exists(onnx_path):
                    logger.info("Found cached model: %s", onnx_path)
                else:
                    logger.info("Exporting model: %s", onnx_path)
                    model = model_obj.get_model().to(model_obj.device)
                    with torch.inference_mode():
                        inputs = model_obj.get_sample_input(1, 512, 512)
                        fp32_inputs = tuple(
                            [
                                (tensor.to(torch.float32) if tensor.dtype == torch.float16 else tensor)
                                for tensor in inputs
                            ]
                        )

                        torch.onnx.export(
                            model,
                            fp32_inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=self.onnx_opset,
                            do_constant_folding=True,
                            input_names=model_obj.get_input_names(),
                            output_names=model_obj.get_output_names(),
                            dynamic_axes=model_obj.get_dynamic_axes(),
                        )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()

                # Optimize onnx
                logger.info("Generating optimized model: %s", onnx_opt_path)
                model_obj.optimize_ort(onnx_path, onnx_opt_path, to_fp16=fp16)

        for model_name in models:
            engine_path = Engines.get_engine_path(engine_dir, model_name, profile_id)
            engine = Engine(engine_path, self.provider, device_id=device_id, enable_cuda_graph=enable_cuda_graph)
            logger.info("%s options for %s: %s", self.provider, model_name, engine.provider_options)
            self.engines[model_name] = engine

    def get_engine(self, model_name):
        return self.engines[model_name]


# -----------------------------------------------------------------------------------------------------
# Utilities for TensorRT EP
# -----------------------------------------------------------------------------------------------------


class OrtTensorrtEngine(CudaSession):
    def __init__(self, engine_path, device_id, onnx_path, fp16, input_profile, workspace_size, enable_cuda_graph):
        self.engine_path = engine_path
        self.ort_trt_provider_options = self.get_tensorrt_provider_options(
            input_profile,
            workspace_size,
            fp16,
            device_id,
            enable_cuda_graph,
        )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        print("creating TRT EP session for ", onnx_path)
        ort_session = ort.InferenceSession(
            onnx_path,
            session_options,
            providers=[
                ("TensorrtExecutionProvider", self.ort_trt_provider_options),
            ],
        )
        print("created TRT EP session for ", onnx_path)

        device = torch.device("cuda", device_id)
        super().__init__(ort_session, device, enable_cuda_graph)

    def get_tensorrt_provider_options(self, input_profile, workspace_size, fp16, device_id, enable_cuda_graph):
        trt_ep_options = {
            "device_id": device_id,
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_timing_cache_enable": True,
            "trt_detailed_build_log": True,
            "trt_engine_cache_path": self.engine_path,
        }

        if enable_cuda_graph:
            trt_ep_options["trt_cuda_graph_enable"] = True

        if workspace_size > 0:
            trt_ep_options["trt_max_workspace_size"] = workspace_size

        if input_profile:
            min_shapes = []
            max_shapes = []
            opt_shapes = []
            for name, profile in input_profile.items():
                assert isinstance(profile, list) and len(profile) == 3
                min_shape = profile[0]
                opt_shape = profile[1]
                max_shape = profile[2]
                assert len(min_shape) == len(opt_shape) and len(opt_shape) == len(max_shape)

                min_shapes.append(f"{name}:" + "x".join([str(x) for x in min_shape]))
                opt_shapes.append(f"{name}:" + "x".join([str(x) for x in opt_shape]))
                max_shapes.append(f"{name}:" + "x".join([str(x) for x in max_shape]))

            trt_ep_options["trt_profile_min_shapes"] = ",".join(min_shapes)
            trt_ep_options["trt_profile_max_shapes"] = ",".join(max_shapes)
            trt_ep_options["trt_profile_opt_shapes"] = ",".join(opt_shapes)

        logger.info("trt_ep_options=%s", trt_ep_options)

        return trt_ep_options


def get_onnx_path(model_name, onnx_dir, opt=True):
    return os.path.join(onnx_dir, model_name + (".opt" if opt else "") + ".onnx")


def get_engine_path(engine_dir, model_name, profile_id):
    return os.path.join(engine_dir, model_name + profile_id)


def has_engine_file(engine_path):
    if os.path.isdir(engine_path):
        children = os.scandir(engine_path)
        for entry in children:
            if entry.is_file() and entry.name.endswith(".engine"):
                return True
    return False


def get_work_space_size(model_name, max_workspace_size):
    gibibyte = 2**30
    workspace_size = 4 * gibibyte if model_name == "clip" else max_workspace_size
    if workspace_size == 0:
        _, free_mem, _ = cudart.cudaMemGetInfo()
        # The following logic are adopted from TensorRT demo diffusion.
        if free_mem > 6 * gibibyte:
            workspace_size = free_mem - 4 * gibibyte
    return workspace_size


def build_engines(
    models,
    engine_dir,
    onnx_dir,
    onnx_opset,
    opt_image_height,
    opt_image_width,
    opt_batch_size=1,
    force_engine_rebuild=False,
    static_batch=False,
    static_image_shape=True,
    max_workspace_size=0,
    device_id=0,
    enable_cuda_graph=False,
):
    if force_engine_rebuild:
        if os.path.isdir(onnx_dir):
            logger.info("Remove existing directory %s since force_engine_rebuild is enabled", onnx_dir)
            shutil.rmtree(onnx_dir)
        if os.path.isdir(engine_dir):
            logger.info("Remove existing directory %s since force_engine_rebuild is enabled", engine_dir)
            shutil.rmtree(engine_dir)

    if not os.path.isdir(engine_dir):
        os.makedirs(engine_dir)

    if not os.path.isdir(onnx_dir):
        os.makedirs(onnx_dir)

    # Export models to ONNX
    for model_name, model_obj in models.items():
        profile_id = model_obj.get_profile_id(
            opt_batch_size, opt_image_height, opt_image_width, static_batch, static_image_shape
        )
        engine_path = get_engine_path(engine_dir, model_name, profile_id)
        if not has_engine_file(engine_path):
            onnx_path = get_onnx_path(model_name, onnx_dir, opt=False)
            onnx_opt_path = get_onnx_path(model_name, onnx_dir)
            if not os.path.exists(onnx_opt_path):
                if not os.path.exists(onnx_path):
                    logger.info(f"Exporting model: {onnx_path}")
                    model = model_obj.get_model()
                    with torch.inference_mode(), torch.autocast("cuda"):
                        inputs = model_obj.get_sample_input(opt_batch_size, opt_image_height, opt_image_width)
                        torch.onnx.export(
                            model,
                            inputs,
                            onnx_path,
                            export_params=True,
                            opset_version=onnx_opset,
                            do_constant_folding=True,
                            input_names=model_obj.get_input_names(),
                            output_names=model_obj.get_output_names(),
                            dynamic_axes=model_obj.get_dynamic_axes(),
                        )
                    del model
                    torch.cuda.empty_cache()
                    gc.collect()
                else:
                    logger.info("Found cached model: %s", onnx_path)

                # Optimize onnx
                if not os.path.exists(onnx_opt_path):
                    logger.info("Generating optimizing model: %s", onnx_opt_path)
                    model_obj.optimize_trt(onnx_path, onnx_opt_path)
                else:
                    logger.info("Found cached optimized model: %s", onnx_opt_path)

    built_engines = {}
    for model_name, model_obj in models.items():
        profile_id = model_obj.get_profile_id(
            opt_batch_size, opt_image_height, opt_image_width, static_batch, static_image_shape
        )

        engine_path = get_engine_path(engine_dir, model_name, profile_id)
        onnx_opt_path = get_onnx_path(model_name, onnx_dir)

        if not has_engine_file(engine_path):
            logger.info(
                "Building TensorRT engine for %s from %s to %s. It can take a while to complete...",
                model_name,
                onnx_opt_path,
                engine_path,
            )
        else:
            logger.info("Reuse cached TensorRT engine in directory %s", engine_path)

        input_profile = model_obj.get_input_profile(
            opt_batch_size,
            opt_image_height,
            opt_image_width,
            static_batch=static_batch,
            static_image_shape=static_image_shape,
        )

        engine = OrtTensorrtEngine(
            engine_path,
            device_id,
            onnx_opt_path,
            fp16=True,
            input_profile=input_profile,
            workspace_size=get_work_space_size(model_name, max_workspace_size),
            enable_cuda_graph=enable_cuda_graph,
        )

        built_engines[model_name] = engine

    return built_engines


def run_engine(engine, feed_dict):
    return engine.infer(feed_dict)


# -----------------------------------------------------------------------------------------------------
# Utilities for both CUDA and TensorRT EP
# -----------------------------------------------------------------------------------------------------

class StableDiffusionPipelineMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def load_models(self):
        #self.embedding_dim = self.text_encoder.config.hidden_size
        assert self.pipeline_info.clip_embedding_dim() == self.text_encoder.config.hidden_size
        
        stages = self.pipeline_info.stages()
        if "clip" in stages:
            self.models["clip"] = CLIP(
                self.pipeline_info,
                self.text_encoder,
                device=self.torch_device,
                max_batch_size=self.max_batch_size,
                clip_skip=0,
            )

        # if "clip2" in stages:
        #     self.models["clip2"] = CLIPWithProj(
        #         self.pipeline_info,
        #         self.text_encoder_2,
        #         device=self.torch_device,
        #         max_batch_size=self.max_batch_size,
        #         clip_skip=0,
        #     )

        if "unet" in stages:
            self.models["unet"] = UNet(
                self.pipeline_info,
                self.unet,
                device=self.torch_device,
                fp16=True,
                max_batch_size=self.max_batch_size,
                unet_dim=(9 if self.pipeline_info.is_inpaint() else 4),
            )

        # if "unetxl" in stages:
        #     self.models["unetxl"] = UNetXL(
        #         self.pipeline_info,
        #         self.unet,
        #         device=self.torch_device,
        #         fp16=True,
        #         max_batch_size=self.max_batch_size,
        #         unet_dim=4,
        #         time_dim=(5 if pipeline_info.is_sd_xl_refiner() else 6),
        #     )

        # VAE Decoder
        if "vae" in stages:
            self.models["vae"] = VAE(
                self.pipeline_info,
                self.vae,
                device=self.torch_device,
                max_batch_size=self.max_batch_size,
            )

    def encode_prompt(self, clip_engine, prompt, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
                prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds`. instead. If not defined, one has to pass `negative_prompt_embeds`. instead.
                Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
        """
        # Tokenize prompt
        text_input_ids = (
            self.tokenizer(
                prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.torch_device)
        )

        # NOTE: output tensor for CLIP must be cloned because it will be overwritten when called again for negative prompt
        text_embeddings = run_engine(clip_engine, {"input_ids": text_input_ids})["text_embeddings"].clone()

        # Tokenize negative prompt
        uncond_input_ids = (
            self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            .input_ids.type(torch.int32)
            .to(self.torch_device)
        )

        uncond_embeddings = run_engine(clip_engine, {"input_ids": uncond_input_ids})["text_embeddings"]

        # Concatenate the unconditional and text embeddings into a single batch to avoid doing two forward passes for classifier free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings]).to(dtype=torch.float16)

        return text_embeddings


    def denoise_latent(self, unet_engine, latents, text_embeddings, timesteps=None, mask=None, masked_image_latents=None, timestep_fp16=False):
        if not isinstance(timesteps, torch.Tensor):
            timesteps = self.scheduler.timesteps
            
        for _step_index, timestep in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, timestep)
            if isinstance(mask, torch.Tensor):
                latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)

            # Predict the noise residual
            timestep_float = timestep.to(torch.float16) if timestep_fp16 else timestep.to(torch.float32)

            noise_pred = run_engine(
                unet_engine,
                {"sample": latent_model_input, "timestep": timestep_float, "encoder_hidden_states": text_embeddings},
            )["latent"]

            # Perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, timestep, latents).prev_sample

        latents = 1.0 / 0.18215 * latents
        return latents

    def decode_latent(self, vae_engine, latents):
        images = run_engine(vae_engine, {"latent": latents})["images"]
        images = (images / 2 + 0.5).clamp(0, 1)
        return images.cpu().permute(0, 2, 3, 1).float().numpy()
    
    def set_cached_folder(self, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs):
        from diffusers.utils import DIFFUSERS_CACHE
        from huggingface_hub import snapshot_download

        cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        revision = kwargs.pop("revision", None)

        self.cached_folder = (
            pretrained_model_name_or_path
            if os.path.isdir(pretrained_model_name_or_path)
            else snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
            )
        )