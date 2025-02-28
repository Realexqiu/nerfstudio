"""
Bruisefacto DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type

import torch
import random
from pathlib import Path
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig


class MyEvalDataset(Dataset):
    def __init__(self, datamanager):
        # store reference to the datamanager and any needed config
        self.datamanager = datamanager
        self.num_images = len(datamanager.eval_dataset)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        """
        Return (camera, batch) for the eval image at 'index'.
        """ 
        # 1) We'll do something similar to your 'next_eval_image'
        dm = self.datamanager
        data = self.datamanager.cached_eval[index].copy()
        data["image"] = data["image"].to(dm.device)

        camera = dm.eval_dataset.cameras[index : index+1].to(dm.device)

        # Load bruise and strawberry mask paths
        yolo_masks_dir = Path(dm.config.data) / "yolo_masks"
        yolo_mask_file_name = f"frame_{index:05d}_bruise_mask.png"
        yolo_mask_path = yolo_masks_dir / yolo_mask_file_name

        strawbery_masks_dir = Path(dm.config.data) / "grounded_sam2_masks"
        strawberry_mask_file_name = f"frame_{index:05d}_strawberry_mask.png"
        strawberry_mask_path = strawbery_masks_dir / strawberry_mask_file_name

        # If bruise mask path exists save mask as tensor and add to data batch
        if yolo_mask_path.exists():
            yolo_mask = Image.open(yolo_mask_path).convert("L")  # Convert mask to grayscale
            yolo_mask_tensor = transforms.ToTensor()(yolo_mask).to(dm.device)
            data[dm.config.bruise_mask_key] = yolo_mask_tensor
        else:
            # Handle missing masks by creating an empty (zeroed) mask
            empty_mask = torch.zeros((1, data["image"].shape[-2], data["image"].shape[-1]), device=dm.device)
            data[dm.config.bruise_mask_key] = empty_mask

        # If strawberry mask path exists save mask as tensor and add to data batch
        if strawberry_mask_path.exists():
            strawberry_mask = Image.open(strawberry_mask_path).convert("L")  # Convert mask to grayscale
            strawberry_mask_tensor = transforms.ToTensor()(strawberry_mask).to(dm.device)
            data[dm.config.strawberry_mask_key] = strawberry_mask_tensor
        else:
            # Handle missing masks by creating an empty (zeroed) mask
            empty_mask = torch.zeros((1, data["image"].shape[-2], data["image"].shape[-1]), device=dm.device)
            data[dm.config.strawberry_mask_key] = empty_mask

        return camera, data
    
@dataclass
class BruisefactoDatamanagerConfig(FullImageDatamanagerConfig):
    """Bruisefacto DataManager Config

    Add your custom datamanager config parameters here.
    """
    bruise_mask_key: str = "bruise_mask" # Key to use for yolo mask in batch
    strawberry_mask_key: str = "strawberry_mask" # Key to use for strawberry mask in batch

    _target: Type = field(default_factory=lambda: BruisefactoDatamanager)

class BruisefactoDatamanager(FullImageDatamanager):
    """Bruisefacto DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: BruisefactoDatamanagerConfig # type: ignore

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 3) Create a dataset object
        self.eval_loader = MyEvalDataset(self)
        
        # 4) Wrap it in a data loader (batch_size=1, no shuffle)

        self.fixed_ind_eval_loader = DataLoader(self.eval_loader, batch_size=1, shuffle=False, collate_fn=single_item_collate_fn)
    
        # self.train_unseen_cameras = self.sample_train_cameras

    @property
    def fixed_indices_eval_dataloader(self):
        """Read-only property that returns our private loader."""
        return self.fixed_ind_eval_loader

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch with YOLO bruise masks.

        Returns a Camera instead of RayBundle for consistency with splatfacto.
        """
        # Pop the next unseen camera index
        image_idx = self.train_unseen_cameras.pop(0)
        # Re-populate the unseen cameras list if exhausted
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = self.sample_train_cameras()

        # Get data from cached training dataset
        data = self.cached_train[image_idx]
        data = data.copy()  # Prevent mutations to cached dictionary
        data["image"] = data["image"].to(self.device)

        # Ensure the camera dimensions match expectations
        assert len(self.train_cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx

        if self.config.data is None:
            raise ValueError("Config data path is None")

        # Load bruise and strawberry mask paths
        yolo_masks_dir = Path(self.config.data) / "yolo_masks"  
        yolo_mask_file_name = f"frame_{image_idx:05d}_bruise_mask.png"
        yolo_mask_path = yolo_masks_dir / yolo_mask_file_name

        strawbery_masks_dir = Path(self.config.data) / "grounded_sam2_masks"
        strawberry_mask_file_name = f"frame_{image_idx:05d}_strawberry_mask.png"
        strawberry_mask_path = strawbery_masks_dir / strawberry_mask_file_name

        # If bruise mask path exists save mask as tensor and add to data batch
        if yolo_mask_path.exists():
            yolo_mask = Image.open(yolo_mask_path).convert("L")  # Convert mask to grayscale
            yolo_mask_tensor = transforms.ToTensor()(yolo_mask).to(self.device)
            data[self.config.bruise_mask_key] = yolo_mask_tensor
        else:
            # Handle missing masks by creating an empty (zeroed) mask
            empty_mask = torch.zeros((1, data["image"].shape[-2], data["image"].shape[-1]), device=self.device)
            data[self.config.bruise_mask_key] = empty_mask

        # If strawberry mask path exists save mask as tensor and add to data batch
        if strawberry_mask_path.exists():
            strawberry_mask = Image.open(strawberry_mask_path).convert("L")  # Convert mask to grayscale
            strawberry_mask_tensor = transforms.ToTensor()(strawberry_mask).to(self.device)
            data[self.config.strawberry_mask_key] = strawberry_mask_tensor
        else:
            # Handle missing masks by creating an empty (zeroed) mask
            empty_mask = torch.zeros((1, data["image"].shape[-2], data["image"].shape[-1]), device=self.device)
            data[self.config.strawberry_mask_key] = empty_mask

        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next single evaluation image (camera + data)."""
        # 1) Pick a random camera index from the unseen list
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # If we've exhausted the unseen list, re-populate
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]

        # Retrieve the camera + data from cache
        data = self.cached_eval[image_idx]
        data = data.copy()  # avoid mutating the cache
        data["image"] = data["image"].to(self.device)

        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension."
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)

        # Load bruise and strawberry mask paths
        yolo_masks_dir = Path(self.config.data) / "yolo_masks" # type: ignore
        yolo_mask_file_name = f"frame_{image_idx:05d}_bruise_mask.png"
        yolo_mask_path = yolo_masks_dir / yolo_mask_file_name

        strawbery_masks_dir = Path(self.config.data) / "grounded_sam2_masks" # type: ignore
        strawberry_mask_file_name = f"frame_{image_idx:05d}_strawberry_mask.png"
        strawberry_mask_path = strawbery_masks_dir / strawberry_mask_file_name

        # If bruise mask path exists save mask as tensor and add to data batch
        if yolo_mask_path.exists():
            yolo_mask = Image.open(yolo_mask_path).convert("L")  # Convert mask to grayscale
            yolo_mask_tensor = transforms.ToTensor()(yolo_mask).to(self.device)
            data[self.config.bruise_mask_key] = yolo_mask_tensor
        else:
            # Handle missing masks by creating an empty (zeroed) mask
            empty_mask = torch.zeros((1, data["image"].shape[-2], data["image"].shape[-1]), device=self.device)
            data[self.config.bruise_mask_key] = empty_mask

        # If strawberry mask path exists save mask as tensor and add to data batch
        if strawberry_mask_path.exists():
            strawberry_mask = Image.open(strawberry_mask_path).convert("L")  # Convert mask to grayscale
            strawberry_mask_tensor = transforms.ToTensor()(strawberry_mask).to(self.device)
            data[self.config.strawberry_mask_key] = strawberry_mask_tensor
        else:
            # Handle missing masks by creating an empty (zeroed) mask
            empty_mask = torch.zeros((1, data["image"].shape[-2], data["image"].shape[-1]), device=self.device)
            data[self.config.strawberry_mask_key] = empty_mask

        return camera, data
    
def single_item_collate_fn(batch):
    # batch is [(camera, data)], just return the single element
    return batch[0]