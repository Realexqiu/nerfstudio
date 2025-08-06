# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processes a video to a nerfstudio compatible dataset and generate both YOLO and SAM masks."""

import shutil
from dataclasses import dataclass
from typing import Literal, Optional
import contextlib
import os

from nerfstudio.process_data import equirect_utils, process_data_utils
from nerfstudio.process_data.colmap_converter_to_nerfstudio_dataset import ColmapConverterToNerfstudioDataset
from nerfstudio.utils.rich_utils import CONSOLE
import ultralytics
from ultralytics import YOLO
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from groundingdino.util.inference import load_model, load_image, predict
from torchvision.ops import box_convert
import torch
import cv2
import numpy as np
from pathlib import Path
import hydra
from hydra import initialize, compose
import hydra.core.global_hydra
from hydra import initialize_config_dir, compose

@dataclass
class VideoToNerfstudioDatasetBruisefacto(ColmapConverterToNerfstudioDataset):
    """Process videos into a nerfstudio dataset.

    This script does the following:

    1. Converts the video into images and downscales them.
    2. Calculates the camera poses for each image using `COLMAP <https://colmap.github.io/>`_.
    3. Applies YOLO segmentation model to images and outputs corresponding bruise masks
    4. Applies Grounded SAM2 to get strawberry masks
    """

    num_frames_target: int = 300
    """Target number of frames to use per video, results may not be exact."""
    percent_radius_crop: float = 1.0
    """Create circle crop mask. The radius is the percent of the image diagonal."""
    matching_method: Literal["exhaustive", "sequential", "vocab_tree"] = "sequential"
    """Feature matching method to use. Vocab tree is recommended for a balance of speed
    and accuracy. Exhaustive is slower but more accurate. Sequential is faster but
    should only be used for videos."""
    random_seed: Optional[int] = None
    """Random seed to select video frames for training set"""
    eval_random_seed: Optional[int] = None
    """Random seed to select video frames for eval set"""
    skip_gsam: bool = False
    """If True, skips gsam mask generation"""
    skip_yolo: bool = False
    """If True, skips yolo mask generation"""

    def main(self) -> None:
        """Process video into a nerfstudio dataset."""
        summary_log = []
        summary_log_eval = []
        # Convert video to images
        if self.camera_type == "equirectangular":
            # create temp images folder to store the equirect and perspective images
            temp_image_dir = self.output_dir / "temp_images"
            temp_image_dir.mkdir(parents=True, exist_ok=True)
            summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
                self.data,
                image_dir=temp_image_dir,
                num_frames_target=self.num_frames_target,
                num_downscales=0,
                crop_factor=(0.0, 0.0, 0.0, 0.0),
                verbose=self.verbose,
                random_seed=self.random_seed,
            )
        else:
            # If we're not dealing with equirects we can downscale in one step.
            summary_log, num_extracted_frames = process_data_utils.convert_video_to_images(
                self.data,
                image_dir=self.image_dir,
                num_frames_target=self.num_frames_target,
                num_downscales=self.num_downscales,
                crop_factor=self.crop_factor,
                verbose=self.verbose,
                image_prefix="frame_train_" if self.eval_data is not None else "frame_",
                keep_image_dir=False,
                random_seed=self.random_seed,
            )
            if self.eval_data is not None:
                summary_log_eval, num_extracted_frames_eval = process_data_utils.convert_video_to_images(
                    self.eval_data,
                    image_dir=self.image_dir,
                    num_frames_target=self.num_frames_target,
                    num_downscales=self.num_downscales,
                    crop_factor=self.crop_factor,
                    verbose=self.verbose,
                    image_prefix="frame_eval_",
                    keep_image_dir=True,
                    random_seed=self.eval_random_seed,
                )
                summary_log += summary_log_eval
                num_extracted_frames += num_extracted_frames_eval

        # Generate planar projections if equirectangular
        if self.camera_type == "equirectangular":
            if self.eval_data is not None:
                raise ValueError("Cannot use eval_data with camera_type equirectangular.")

            perspective_image_size = equirect_utils.compute_resolution_from_equirect(
                self.output_dir / "temp_images", self.images_per_equirect
            )

            equirect_utils.generate_planar_projections_from_equirectangular(
                self.output_dir / "temp_images",
                perspective_image_size,
                self.images_per_equirect,
                crop_factor=self.crop_factor,
            )

            # copy the perspective images to the image directory
            process_data_utils.copy_images(
                self.output_dir / "temp_images" / "planar_projections",
                image_dir=self.output_dir / "images",
                verbose=False,
            )

            # remove the temp_images folder
            shutil.rmtree(self.output_dir / "temp_images", ignore_errors=True)

            self.camera_type = "perspective"

            # # Downscale images
            summary_log.append(process_data_utils.downscale_images(self.image_dir, self.num_downscales, verbose=self.verbose))

        # Create mask
        mask_path = process_data_utils.save_mask(
            image_dir=self.image_dir,
            num_downscales=self.num_downscales,
            crop_factor=(0.0, 0.0, 0.0, 0.0),
            percent_radius=self.percent_radius_crop,
        )
        if mask_path is not None:
            summary_log.append(f"Saved mask to {mask_path}")

        # Run Colmap
        if not self.skip_colmap:
            self._run_colmap(mask_path)

            # Export depth maps
            image_id_to_depth_path, log_tmp = self._export_depth()
            summary_log += log_tmp

            summary_log += self._save_transforms(num_extracted_frames, image_id_to_depth_path, mask_path)

        ## YOLO Segmentation -----------------------------------------------------------------------------------------
        if not self.skip_yolo:
            CONSOLE.log("[bold green]: YOLO Mask Generation Started:")
            # Path to best weights for YOLO segmentation model
            yolo_weights = Path(__file__).parent.parent / 'bruisefacto' / 'bruisefacto' / 'yolo_models' / 'v12_appended_P0.942_R0.895.pt'

            # Apply YOLO segmentation model to each image
                
            bruise_counter = 0
            yolo_model = YOLO(yolo_weights)
            yolo_mask_dir = self.output_dir / "yolo_masks"
            yolo_mask_dir.mkdir(parents=True, exist_ok=True)

            # Iterate through each image in the image directory
            for image_path in self.image_dir.iterdir():
                if image_path.suffix not in [".jpg", ".png"]:
                    continue
                
                # Run YOLO model on image
                bruise_confidence = 0.76
                yolo_results = yolo_model(str(image_path), verbose=False, conf=bruise_confidence)
                yolo_pred = yolo_results[0]

                # Load the original image
                img = cv2.imread(image_path) # type: ignore

                # Create black and white image to save binary mask
                yolo_binary_mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

                # Save mask if YOLO model detected any bruises
                if yolo_pred.masks is not None:

                    # Extract the mask (pred.masks.data contains the binary masks)
                    yolo_masks_list = yolo_pred.masks.data.cpu().numpy()  # Convert to numpy array if needed

                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR (OpenCV) to RGB (matplotlib)

                    # Iterate through each mask in masks list to combine into one mask file (if multiple bruises detected)
                    for yolo_mask in yolo_masks_list:
                        yolo_mask_resized = cv2.resize(yolo_mask, (img.shape[1], img.shape[0]))  # Ensure mask matches image size
                        yolo_binary_mask[yolo_mask_resized > 0.5] = 255
                    
                    bruise_counter += 1

                # Save combined binary mask
                yolo_combined_mask_path = yolo_mask_dir / f"{image_path.stem}_bruise_mask.png"
                cv2.imwrite(str(yolo_combined_mask_path), yolo_binary_mask)

            CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")
            CONSOLE.log("[bold green]:tada: :tada: :tada: YOLO masks successfully generated! :tada: :tada: :tada:")
            CONSOLE.log(f"[bold blue]Found bruises in {bruise_counter} out of {num_extracted_frames} images") # type: ignore
        else:
            CONSOLE.log("[bold green]: YOLO Mask Generation Skipped:")

        ## Grounded SAM2 Mask Segmentation
        if not self.skip_gsam:
            CONSOLE.log("[bold green]: Grounded SAM2 Mask Generation Started:")
            # Grounded SAM Parameters:
            PROMPT = "red strawberry without leaves."  # text query must be lowercased and end with a dot
            SAM2_CHECKPOINT = '/home/alex/Documents/Grounded-SAM-2/checkpoints/sam2.1_hiera_base_plus.pt'
            SAM2_MODEL_CONFIG = '//home/alex/Documents/Grounded-SAM-2/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml'
            G_DINO_CHECKPOINT = '/home/alex/Documents/Grounded-SAM-2/gdino_checkpoints/groundingdino_swint_ogc.pth'
            G_DINO_CONFIG = '/home/alex/Documents/Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py'
            BOX_THRESHOLD = 0.35
            TEXT_THRESHOLD = 0.25
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

            # Directories (assuming these directories already exist)
            sam_mask_dir = self.output_dir / "grounded_sam2_masks"  # Output folder for masks
            sam_mask_dir.mkdir(parents=True, exist_ok=True)

            # Build SAM2 image predictor
            sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
            sam2_predictor = SAM2ImagePredictor(sam2_model)

            # Build GroundingDINO model
            g_model = load_model(model_config_path=G_DINO_CONFIG, model_checkpoint_path=G_DINO_CHECKPOINT, device=DEVICE)

            # Create strawberry counter to check if each frame has detected strawberry
            strawberry_counter = 0
            multi_mask_counter = 0
            multi_masks_path = []

            # Iterate through each image in the image directory
            for image_path in self.image_dir.iterdir():
                if image_path.suffix.lower() not in [".jpg", ".png", ".jpeg"]:
                    continue

                # Load image using GroundingDINO utility (returns original image and one for SAM2)
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull):
                        image_source, img = load_image(str(image_path))
                h, w, _ = image_source.shape

                # Set image for SAM2 predictor
                sam2_predictor.set_image(image_source)

                # Get bounding boxes from GroundingDINO using the text query
                boxes, _, _ = predict(g_model, img, PROMPT, BOX_THRESHOLD, TEXT_THRESHOLD)

                # If no bounding boxes are detected, create an empty binary mask
                if boxes.numel() == 0:
                    final_sam_mask = np.zeros((h, w), dtype=np.uint8)
                else:
                    # Rescale boxes to image dimensions and convert from cxcywh to xyxy
                    boxes = boxes * torch.Tensor([w, h, w, h])
                    input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

                    # Run SAM2 to get masks for the given boxes
                    sam_mask, _, _ = sam2_predictor.predict(point_coords=None, point_labels=None, box=input_boxes, multimask_output=False)

                    # Remove extra dimension if present
                    if sam_mask.ndim == 4:
                        sam_mask = sam_mask.squeeze(1)

                    # Initialize the final binary mask
                    final_sam_mask = np.zeros((h, w), dtype=np.uint8)

                    # If multiple masks are detected, combine them into a single mask
                    if sam_mask.shape[0] > 1:
                        multi_mask_counter += 1
                        multi_masks_path.append(image_path)
                        
                        best_mask = None
                        best_value = -1

                        for mask in sam_mask:
                            mask_resized = cv2.resize(mask, (w, h))
                            binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

                            # Count the number of nonzero pixels in the mask = area
                            area = np.count_nonzero(binary_mask)
                            if area > best_value:
                                best_value = area
                                best_mask = binary_mask
                        
                        # # Iterate over each detected mask
                        # for mask in sam_mask:
                        #     # Resize the mask to match the image dimensions
                        #     mask_resized = cv2.resize(mask, (w, h))
                        #     binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255
                            
                        #     # Apply the mask to the original image
                        #     masked_img = cv2.bitwise_and(image_source, image_source, mask=binary_mask)
                            
                        #     # Count red pixels in the masked area. Since OpenCV uses BGR, the red channel is at index 2.
                        #     red_c, green_c, blue_c = masked_img[:, :, 2], masked_img[:, :, 1], masked_img[:, :, 0]

                        #     # Define a red pixel: red > 127 and red > green and red > blue.
                        #     red_pixels = np.logical_and(red_c > 127, np.logical_and(red_c > green_c, red_c > blue_c))
                        #     red_count = np.sum(red_pixels)
                            
                        #     # Update the best mask if this one has more red pixels.
                        #     if red_count > best_red_count:
                        #         best_red_count = red_count
                        #         best_mask = binary_mask

                        # Return only the mask with the most red pixels, as this is most likely to be the strawberry
                        final_sam_mask = best_mask
                    elif sam_mask.shape[0] == 1:
                        strawberry_counter += 1
                        mask_resized = cv2.resize(sam_mask[0], (w, h))  # Resize if needed
                        final_sam_mask = (mask_resized > 0.5).astype(np.uint8) * 255  # Convert to binary

                # Save the combined binary mask image (naming convention similar to YOLO snippet)
                sam_output_mask_path = sam_mask_dir / f"{image_path.stem}_strawberry_mask.png"
                cv2.imwrite(str(sam_output_mask_path), final_sam_mask) # type: ignore

            CONSOLE.log("[bold green]:tada: :tada: :tada: All DONE :tada: :tada: :tada:")
            CONSOLE.log("[bold green]:tada: :tada: :tada: Grounded SAM2 masks successfully generated! :tada: :tada: :tada:")
            CONSOLE.log(f"[bold blue] {strawberry_counter} frames out of {num_extracted_frames} with strawberries found.")
            CONSOLE.log(f"[bold blue]Found {multi_mask_counter} frames with multiple masks.")
        else:
            CONSOLE.log("[bold green]: Grounded SAM2 Mask Generation Skipped:")

        for summary in summary_log:
            CONSOLE.log(summary)
