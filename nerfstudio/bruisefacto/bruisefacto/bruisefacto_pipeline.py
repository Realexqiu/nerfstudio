"""
Bruisefacto Pipeline
"""

import typing
from dataclasses import dataclass, field
from typing import Literal, Optional, Type
import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Type, Union, Any
from pathlib import Path
import torchvision.utils as vutils
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn

import torch.distributed as dist
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nerfstudio.bruisefacto.bruisefacto.bruisefacto_datamanager import BruisefactoDatamanagerConfig
from nerfstudio.bruisefacto.bruisefacto.bruisefacto import BruisefactoModel, BruisefactoModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)


@dataclass
class BruisefactoPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: BruisefactoPipeline)
    """target class to instantiate"""
    datamanager: DataManagerConfig = BruisefactoDatamanagerConfig()
    """specifies the datamanager config"""
    model: ModelConfig = BruisefactoModelConfig()
    """specifies the model config"""


class BruisefactoPipeline(VanillaPipeline):
    """Bruisefacto Pipeline

    Args:
        config: the pipeline config used to instantiate class
    """

    def __init__(
        self,
        config: BruisefactoPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,)
        self.config = config
        self.test_mode = test_mode
        self.datamanager: DataManager = config.datamanager.setup(
            device=device, test_mode=test_mode, world_size=world_size, local_rank=local_rank
        )

        seed_pts = None
        if (
            hasattr(self.datamanager, "train_dataparser_outputs")
            and "points3D_xyz" in self.datamanager.train_dataparser_outputs.metadata
        ):
            pts = self.datamanager.train_dataparser_outputs.metadata["points3D_xyz"]
            pts_rgb = self.datamanager.train_dataparser_outputs.metadata["points3D_rgb"]
            seed_pts = (pts, pts_rgb)

        self.datamanager.to(device)

        assert self.datamanager.train_dataset is not None, "Missing input dataset"
        self._model = config.model.setup(
            scene_box=self.datamanager.train_dataset.scene_box,
            num_train_data=len(self.datamanager.train_dataset),
            metadata=self.datamanager.train_dataset.metadata,
            device=device,
            grad_scaler=grad_scaler,
            seed_points=seed_pts,
        )
        self.model.to(device)

        self.world_size = world_size
        if world_size > 1:
            self._model = typing.cast(
                BruisefactoModel, DDP(self._model, device_ids=[local_rank], find_unused_parameters=True)
            )
            dist.barrier(device_ids=[local_rank])

