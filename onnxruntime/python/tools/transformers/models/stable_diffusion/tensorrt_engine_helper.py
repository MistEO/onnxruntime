#
# SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# -------------------------------------------------------------------------
# Modified from TensorRT demo diffusion: add pipeline info, refactoring models, use onnxruntime
#
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import gc
import os
import pathlib

import numpy as np
import nvtx
import torch
from cuda import cudart
from diffusion_models import CLIP, VAE, CLIPWithProj, PipelineInfo, UNet, UNetXL, get_tokenizer
from trt_demo.utilities import Engine


class TensorrtEngineHelper:
    """
    Helper class to hide the detail of TensorRT Engine from pipeline.
    """

    def __init__(
        self,
        pipeline_info: PipelineInfo,
        max_batch_size=16,
        hf_token=None,
        device="cuda",
        use_cuda_graph=False,
    ):
        """
        Initializes the TensorRT Engine Helper.

        Args:
            pipeline_info (PipelineInfo):
                Version and Type of pipeline.
            stages (list):
                Ordered sequence of stages. Options: ['vae_encoder', 'clip','unet','vae']
            max_batch_size (int):
                Maximum batch size for dynamic batch engine.
            hf_token (str):
                HuggingFace User Access Token to use for downloading Stable Diffusion model checkpoints.
            use_cuda_graph (bool):
                Use CUDA graph to capture engine execution and then launch inference
        """
        self.pipeline_info = pipeline_info
        self.max_batch_size = max_batch_size
        self.hf_token = hf_token
        self.device = device
        self.use_cuda_graph = use_cuda_graph
        
        self.torch_device = torch.device(device, torch.cuda.current_device())
        self.stages = pipeline_info.stages()
        self.vae_torch_fallback = self.pipeline_info.is_sd_xl()
        self.stream = None

        self.models = {}
        self.torch_models = {}
        self.engine = {}
        self.shared_device_memory = None

    def load_resources(self, image_height, image_width, batch_size):
        err, self.stream = cudart.cudaStreamCreate()

        # Allocate buffers for TensorRT engine bindings
        for model_name, obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue
            self.engine[model_name].allocate_buffers(
                shape_dict=obj.get_shape_dict(batch_size, image_height, image_width), device=self.device
            )

    def teardown(self):
        for engine in self.engine.values():
            del engine

        if self.shared_device_memory:
            cudart.cudaFree(self.shared_device_memory)

        cudart.cudaStreamDestroy(self.stream)
        del self.stream

    def get_cached_model_name(self, model_name):
        if self.pipeline_info.is_inpaint():
            model_name += "_inpaint"
        return model_name

    def get_onnx_path(self, model_name, onnx_dir, opt=True):
        onnx_model_dir = os.path.join(onnx_dir, self.get_cached_model_name(model_name) + (".opt" if opt else ""))
        os.makedirs(onnx_model_dir, exist_ok=True)
        return os.path.join(onnx_model_dir, "model.onnx")

    def get_engine_path(self, model_name, engine_dir):
        return os.path.join(engine_dir, self.get_cached_model_name(model_name) + ".plan")

    def load_engines(
        self,
        engine_dir,
        framework_model_dir,
        onnx_dir,
        onnx_opset,
        opt_batch_size,
        opt_image_height,
        opt_image_width,
        force_export=False,
        force_optimize=False,
        force_build=False,
        static_batch=False,
        static_shape=True,
        enable_refit=False,
        enable_preview=False,
        enable_all_tactics=False,
        timing_cache=None,
        onnx_refit_dir=None,
    ):
        """
        Build and load engines for TensorRT accelerated inference.
        Export ONNX models first, if applicable.

        Args:
            engine_dir (str):
                Directory to write the TensorRT engines.
            framework_model_dir (str):
                Directory to write the framework model ckpt.
            onnx_dir (str):
                Directory to write the ONNX models.
            onnx_opset (int):
                ONNX opset version to export the models.
            opt_batch_size (int):
                Batch size to optimize for during engine building.
            opt_image_height (int):
                Image height to optimize for during engine building. Must be a multiple of 8.
            opt_image_width (int):
                Image width to optimize for during engine building. Must be a multiple of 8.
            force_export (bool):
                Force re-exporting the ONNX models.
            force_optimize (bool):
                Force re-optimizing the ONNX models.
            force_build (bool):
                Force re-building the TensorRT engine.
            static_batch (bool):
                Build engine only for specified opt_batch_size.
            static_shape (bool):
                Build engine only for specified opt_image_height & opt_image_width. Default = True.
            enable_refit (bool):
                Build engines with refit option enabled.
            enable_preview (bool):
                Enable TensorRT preview features.
            enable_all_tactics (bool):
                Enable all tactic sources during TensorRT engine builds.
            timing_cache (str):
                Path to the timing cache to accelerate build or None
            onnx_refit_dir (str):
                Directory containing refit ONNX models.
        """
        # Create directory
        for directory in [engine_dir, onnx_dir]:
            if not os.path.exists(directory):
                print(f"[I] Create directory: {directory}")
                pathlib.Path(directory).mkdir(parents=True)

        if "clip" in self.stages:
            self.models["clip"] = CLIP(
                self.pipeline_info,
                None,  # not loaded yet
                device=self.torch_device,
                max_batch_size=self.max_batch_size,
                clip_skip=0,
            )

        if "clip2" in self.stages:
            self.models["clip2"] = CLIPWithProj(
                self.pipeline_info,
                None,  # not loaded yet
                device=self.torch_device,
                max_batch_size=self.max_batch_size,
                clip_skip=0,
            )

        if "unet" in self.stages:
            self.models["unet"] = UNet(
                self.pipeline_info,
                None,  # not loaded yet
                device=self.torch_device,
                fp16=True,
                max_batch_size=self.max_batch_size,
                unet_dim=(9 if self.pipeline_info.is_inpaint() else 4),
            )

        if "unetxl" in self.stages:
            self.models["unetxl"] = UNetXL(
                self.pipeline_info,
                None,  # not loaded yet
                device=self.torch_device,
                fp16=True,
                max_batch_size=self.max_batch_size,
                unet_dim=4,
                time_dim=(5 if self.pipeline_info.is_sd_xl_refiner() else 6),
            )

        # VAE Decoder
        if "vae" in self.stages:
            self.models["vae"] = VAE(
                self.pipeline_info,
                None,  # not loaded yet
                device=self.torch_device,
                max_batch_size=self.max_batch_size,
            )

            if self.vae_torch_fallback:
                self.torch_models["vae"] = self.models["vae"].load_model(framework_model_dir, self.hf_token)

        # Export models to ONNX
        for model_name, obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue
            engine_path = self.get_engine_path(model_name, engine_dir)
            if force_export or force_build or not os.path.exists(engine_path):
                onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
                onnx_opt_path = self.get_onnx_path(model_name, onnx_dir)
                if force_export or not os.path.exists(onnx_opt_path):
                    if force_export or not os.path.exists(onnx_path):
                        print(f"Exporting model: {onnx_path}")
                        model = obj.load_model(framework_model_dir, self.hf_token)
                        with torch.inference_mode(), torch.autocast("cuda"):
                            inputs = obj.get_sample_input(1, opt_image_height, opt_image_width)
                            torch.onnx.export(
                                model,
                                inputs,
                                onnx_path,
                                export_params=True,
                                opset_version=onnx_opset,
                                do_constant_folding=True,
                                input_names=obj.get_input_names(),
                                output_names=obj.get_output_names(),
                                dynamic_axes=obj.get_dynamic_axes(),
                            )
                        del model
                        torch.cuda.empty_cache()
                        gc.collect()
                    else:
                        print(f"Found cached model: {onnx_path}")

                    # Optimize onnx
                    if force_optimize or not os.path.exists(onnx_opt_path):
                        print(f"Generating optimizing model: {onnx_opt_path}")
                        obj.optimize_trt(onnx_path, onnx_opt_path)
                    else:
                        print(f"Found cached optimized model: {onnx_opt_path} ")

        # Build TensorRT engines
        for model_name, obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue
            engine_path = self.get_engine_path(model_name, engine_dir)
            engine = Engine(engine_path)
            onnx_path = self.get_onnx_path(model_name, onnx_dir, opt=False)
            onnx_opt_path = self.get_onnx_path(model_name, onnx_dir)

            if force_build or not os.path.exists(engine.engine_path):
                engine.build(
                    onnx_opt_path,
                    fp16=True,
                    input_profile=obj.get_input_profile(
                        opt_batch_size,
                        opt_image_height,
                        opt_image_width,
                        static_batch,
                        static_shape,
                    ),
                    enable_refit=enable_refit,
                    enable_preview=enable_preview,
                    enable_all_tactics=enable_all_tactics,
                    timing_cache=timing_cache,
                    update_output_names=None,
                )
            self.engine[model_name] = engine

        # Load TensorRT engines
        for model_name, obj in self.models.items():
            if model_name == "vae" and self.vae_torch_fallback:
                continue
            self.engine[model_name].load()
            if onnx_refit_dir:
                onnx_refit_path = self.get_onnx_path(model_name, onnx_refit_dir)
                if os.path.exists(onnx_refit_path):
                    self.engine[model_name].refit(onnx_opt_path, onnx_refit_path)
        
    def max_device_memory(self):
        max_device_memory = 0
        for _model_name, engine in self.engine.items():
            max_device_memory = max(max_device_memory, engine.engine.device_memory_size)
        return max_device_memory

    def activate_engines(self, shared_device_memory=None):
        if shared_device_memory is None:
            max_device_memory = self.max_device_memory()
            _, shared_device_memory = cudart.cudaMalloc(max_device_memory)
        self.shared_device_memory = shared_device_memory
        # Load and activate TensorRT engines
        for engine in self.engine.values():
            engine.activate(reuse_device_memory=self.shared_device_memory)

    def run_engine(self, model_name, feed_dict):
        engine = self.engine[model_name]
        return engine.infer(feed_dict, self.stream, use_cuda_graph=self.use_cuda_graph)
        
    def vae_decode(self, latents):
        if self.vae_torch_fallback:
            latents = latents.to(dtype=torch.float32)
            self.torch_models["vae"] = self.torch_models["vae"].to(dtype=torch.float32)
            images = self.torch_models["vae"](latents)["sample"]
        else:
            images = self.run_engine("vae", {"latent": latents})["images"]
            
        return images