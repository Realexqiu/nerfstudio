"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""

from __future__ import annotations

from bruisefacto.bruisefacto_datamanager import BruisefactoDatamanagerConfig
from bruisefacto.bruisefacto import BruisefactoModelConfig
from bruisefacto.bruisefacto_pipeline import BruisefactoPipelineConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification


bruisefacto_method = MethodSpecification(
    config=TrainerConfig(
        method_name="bruisefacto",  # TODO: rename to your own model
        steps_per_eval_batch=200,
        steps_per_save=2000,
        max_num_iterations=15000,
        mixed_precision=False,
        pipeline=BruisefactoPipelineConfig(
            datamanager=BruisefactoDatamanagerConfig(
                dataparser=NerfstudioDataParserConfig(load_3D_points=True),
                cache_images_type="uint8",
            ),
            model=BruisefactoModelConfig(
                # cull_alpha_thresh=0.005,
                # densify_grad_thresh=0.0005,
                eval_num_rays_per_chunk=1 << 15
            ),
        ),
        
        optimizers={
            "bruise": {
                "optimizer": AdamOptimizerConfig(lr=0.01, eps=1e-15),
                "scheduler": None,
            },
            "strawberry": {
                "optimizer": AdamOptimizerConfig(lr=0.1, eps=1e-15),
                "scheduler": None,
            },
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1.6e-6, max_steps=30000)
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), 
                "scheduler": None
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-7, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=5e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-4, max_steps=30000, warmup_steps=1000, lr_pre_warmup=0),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Bruisefacto for strawberry bruising analysis",
)
