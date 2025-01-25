"""
Bruisefacto DataManager
"""

from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type, Union

import torch
import random
from pathlib import Path
from torchvision.transforms import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader


from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig

import inspect

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
        Reuse logic from your 'next_eval_image()' or do something similar.
        """
        # 1) We'll do something similar to your 'next_eval_image'
        data = self.datamanager.cached_eval[index].copy()
        data["image"] = data["image"].to(self.datamanager.device)

        camera = self.datamanager.eval_dataset.cameras[index : index+1].to(self.datamanager.device)

        # 2) Attach bruise mask from disk
        yolo_masks_dir = Path(self.datamanager.config.data) / "yolo_masks"
        mask_file_name = f"frame_{index:05d}_bruise_mask.png"
        mask_path = yolo_masks_dir / mask_file_name

        if mask_path.exists():
            mask_img = Image.open(mask_path).convert("L")
            bruise_mask = transforms.ToTensor()(mask_img).to(self.datamanager.device)
        else:
            c, h, w = data["image"].shape
            bruise_mask = torch.zeros((1, h, w), device=self.datamanager.device)
        data[self.datamanager.config.bruise_mask_key] = bruise_mask

        return camera, data
    
@dataclass
class BruisefactoDatamanagerConfig(FullImageDatamanagerConfig):
    """Bruisefacto DataManager Config

    Add your custom datamanager config parameters here.
    """
    bruise_mask_key: str = "bruise_mask" # Key to use for yolo mask in batch

    _target: Type = field(default_factory=lambda: BruisefactoDatamanager)


class BruisefactoDatamanager(FullImageDatamanager):
    """Bruisefacto DataManager

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: BruisefactoDatamanagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 3) Create a dataset object
        self.eval_dataset_for_loader = MyEvalDataset(self)
        
        # 4) Wrap it in a data loader (batch_size=1, no shuffle)


        self._fixed_indices_eval_dataloader = DataLoader(
            self.eval_dataset_for_loader,
            batch_size=1,
            shuffle=False,
            collate_fn=single_item_collate_fn,
        )
    
        # self.train_unseen_cameras = self.sample_train_cameras

    @property
    def fixed_indices_eval_dataloader(self):
        """Read-only property that returns our private loader."""
        return self._fixed_indices_eval_dataloader

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

        # Add YOLO bruise mask to the data
        bruise_mask_key = self.config.bruise_mask_key  # Key for the mask in the batch

        if self.config.data is None:
            raise ValueError("Config data path is None")

        yolo_masks_dir = Path(self.config.data) / "yolo_masks"  # Path to YOLO masks directory
        mask_file_name = f"frame_{image_idx:05d}_bruise_mask.png"
        mask_path = yolo_masks_dir / mask_file_name

        if mask_path.exists():
            mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
            mask_tensor = transforms.ToTensor()(mask).to(self.device)
            data[bruise_mask_key] = mask_tensor
        else:
            # Handle missing masks by creating an empty (zeroed) mask
            print(f"Mask not found for: {mask_path}. Creating a default empty mask.")
            empty_mask = torch.zeros((1, data["image"].shape[-2], data["image"].shape[-1]), device=self.device)
            data[bruise_mask_key] = empty_mask

        # import pdb; pdb.set_trace()  # Debugging breakpoint

        return camera, data

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next single evaluation image (camera + data)."""
        # 1) Pick a random camera index from the unseen list
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # If we've exhausted the unseen list, re-populate
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]

        # 2) Retrieve the camera + data from cache
        data = self.cached_eval[image_idx]
        data = data.copy()  # avoid mutating the cache
        data["image"] = data["image"].to(self.device)

        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension."
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)

        # 3) Load bruise mask from disk (similar to how you do in next_train)
        yolo_masks_dir = Path(self.config.data) / "yolo_masks"
        mask_file_name = f"frame_{image_idx:05d}_bruise_mask.png"
        mask_path = yolo_masks_dir / mask_file_name

        if mask_path.exists():
            mask_img = Image.open(mask_path).convert("L")
            bruise_mask = transforms.ToTensor()(mask_img).to(self.device)  # shape [1, H, W]
        else:
            # If not found, build an empty mask
            print(f"[Eval] Mask not found for {mask_path}, using an empty mask.")
            empty_mask = torch.zeros((1, data["image"].shape[-2], data["image"].shape[-1]), device=self.device)
            data[self.config.bruise_mask_key] = empty_mask

        data["bruise_mask"] = bruise_mask
        print("bruisefacto_datamanager.py: next_eval_image() called")

        return camera, data
    
def single_item_collate_fn(batch):
    # batch is [(camera, data)], just return the single element
    return batch[0]