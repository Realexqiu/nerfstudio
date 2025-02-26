"""
Modified model for Splatfacto with 2D bruising distillation
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union, Any
import inspect

import cv2
import torch
from torch import Tensor
from gsplat.rendering import rasterization
from pytorch_msssim import SSIM
from torch.nn import Parameter
import torch.nn.functional as F

from nerfstudio.model_components.lib_bilagrid import BilateralGrid, total_variation_loss
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox

from nerfstudio.model_components import renderers
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig  # for subclassing Nerfacto model
from nerfstudio.utils.misc import torch_compile
from einops import repeat
from gsplat.strategy import DefaultStrategy

@dataclass
class BruisefactoModelConfig(SplatfactoModelConfig):
    """Bruisefacto Model Configuration.

    Add your custom model config parameters here.
    """
    
    _target: Type = field(default_factory=lambda: BruisefactoModel) 


class BruisefactoModel(SplatfactoModel):
    """Bruisefacto Model."""

    config: BruisefactoModelConfig # type: ignore

    @property
    def bruise(self):
        return self.gauss_params["bruise"]
    
    @property
    def strawberry(self):
        return self.gauss_params["strawberry"]

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        param_groups = {
            name: [self.gauss_params[name]]
            for name in ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
        }
        param_groups["bruise"] = [self.gauss_params["bruise"]]
        param_groups["strawberry"] = [self.gauss_params["strawberry"]]
        return param_groups

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "means" in dict:
            # For backwards compatibility, remap names of parameters from means->gauss_params.means for old checkpoints
            param_list = ["means", "scales", "quats", "features_dc", "features_rest", "opacities"]
            param_list.append("bruise")
            param_list.append("strawberry")

            for p in param_list:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.means"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((self.config.num_random, 3)) - 0.5) * self.config.random_scale)

        distances, _ = self.k_nearest_sklearn(means.data, 3)
        distances = torch.from_numpy(distances)

        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        scales = torch.nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        num_points = means.shape[0]
        quats = torch.nn.Parameter(random_quat_tensor(num_points))
        dim_sh = num_sh_bases(self.config.sh_degree)

        # Randomly initialize strawberry and bruise masks
        bruise = torch.nn.Parameter(torch.rand(means.shape[0], 1))
        strawberry = torch.nn.Parameter(torch.rand(means.shape[0], 1))

        # We can have colors without points.
        if (self.seed_points is not None and not self.config.random_init and self.seed_points[1].shape[0] > 0):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            features_dc = torch.nn.Parameter(torch.rand(num_points, 3))
            features_rest = torch.nn.Parameter(torch.zeros((num_points, dim_sh - 1, 3)))

        opacities = torch.nn.Parameter(torch.logit(0.1 * torch.ones(num_points, 1)))
        self.gauss_params = torch.nn.ParameterDict({
            "means": means,
            "scales": scales,
            "quats": quats,
            "features_dc": features_dc,
            "features_rest": features_rest,
            "opacities": opacities,
            "bruise": bruise, # type: ignore
            "strawberry": strawberry # type: ignore
        })

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        if self.config.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                num=self.num_train_data,
                grid_X=self.config.grid_shape[0],
                grid_Y=self.config.grid_shape[1],
                grid_W=self.config.grid_shape[2],
            )

        # Strategy for GS densification
        self.strategy = DefaultStrategy(
            prune_opa=self.config.cull_alpha_thresh,
            grow_grad2d=self.config.densify_grad_thresh,
            grow_scale3d=self.config.densify_size_thresh,
            grow_scale2d=self.config.split_screen_size,
            prune_scale3d=self.config.cull_scale_thresh,
            prune_scale2d=self.config.cull_screen_size,
            refine_scale2d_stop_iter=self.config.stop_screen_size_at,
            refine_start_iter=self.config.warmup_length,
            refine_stop_iter=self.config.stop_split_at,
            reset_every=self.config.reset_alpha_every * self.config.refine_every,
            refine_every=self.config.refine_every,
            pause_refine_after_reset=self.num_train_data + self.config.refine_every,
            absgrad=self.config.use_absgrad,
            revised_opacity=False,
            verbose=True,
        )
        self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)

    def get_background(self):
        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)
        return background
    
    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, Any]):
        """
        Called by the pipeline for each image at eval time. 
        Returns:
            metrics_dict, images_dict
        """
        # -- 1) First let the base class fill out standard metrics & images (PSNR, depth, etc.)
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)

        # -- 2) Add bruise metrics if bruise is in batch
        if "bruise" in outputs and "bruise_mask" in batch:
            pred_bruise = outputs["bruise"]   # shape [H, W] or [H, W, 1]
            gt_bruise   = batch["bruise_mask"] # shape [H, W] or [1, H, W], etc.

            if pred_bruise.shape[:2] != gt_bruise.shape[:2]:
                gt_bruise = resize_bruise_mask(gt_bruise, pred_bruise.shape[:2])

            iou_value = compute_bruise_iou(pred_bruise, gt_bruise)
            metrics_dict["bruise_iou"] = iou_value

            # Add PSNR for Bruise
            bruise_psnr = compute_psnr(pred_bruise, gt_bruise)
            metrics_dict["bruise_psnr"] = bruise_psnr

        if "strawberry" in outputs and "strawberry_mask" in batch:
            pred_strawberry = outputs["strawberry"]   # shape [H, W] or [H, W, 1]
            gt_strawberry   = batch["strawberry_mask"] # shape [H, W] or [1, H, W], etc.

            if pred_strawberry.shape[:2] != gt_strawberry.shape[:2]:
                gt_strawberry = resize_bruise_mask(gt_strawberry, pred_strawberry.shape[:2])
            
            return metrics_dict, images_dict
    
    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a camera and returns a dictionary of outputs.

        Args:
            camera: The camera(s) for which output images are rendered. It should have
            all the needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
            optimized_camera_to_world = camera.camera_to_worlds

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()), int(camera.height.item()), self.background_color
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
            bruise_crop = self.bruise[crop_ids]
            strawberry_crop = self.strawberry[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats
            bruise_crop = self.bruise
            strawberry_crop = self.strawberry

        colors_crop = torch.cat((features_dc_crop[:, None, :], features_rest_crop), dim=1)

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        viewmat = get_viewmat(optimized_camera_to_world)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        # apply the compensation of screen space blurring to gaussians
        if self.config.rasterize_mode not in ["antialiased", "classic"]:
            raise ValueError("Unknown rasterize_mode: %s", self.config.rasterize_mode)

        if self.config.output_depth_during_training or not self.training:
            render_mode = "RGB+ED"
        else:
            render_mode = "RGB"

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(self.step // self.config.sh_degree_interval, self.config.sh_degree)
        else:
            colors_crop = torch.sigmoid(colors_crop).squeeze(1)  # [N, 1, 3] -> [N, 3]
            sh_degree_to_use = None

        render, alpha, self.info = rasterization(
            means=means_crop,
            quats=quats_crop,  # rasterization does normalization internally
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            colors=colors_crop,
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        # Perform bruise mask rendering
        bruise_mask, _, _ = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            # replace colors with 3x bruise_crop
            colors = repeat(bruise_crop, 'n 1 -> n 1 3'),
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=0,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
        )
        bruise_mask = bruise_mask[..., 0:1]  # Use the first channel for the mask

        # Perform strawberry mask rendering
        strawberry_mask, _, _ = rasterization(
            means=means_crop,
            quats=quats_crop,
            scales=torch.exp(scales_crop),
            opacities=torch.sigmoid(opacities_crop).squeeze(-1),
            # replace colors with 3x strawberry_crop
            colors = repeat(strawberry_crop, 'n 1 -> n 1 3'),
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            render_mode=render_mode,
            sh_degree=0,
            sparse_grad=False,
            absgrad=self.strategy.absgrad,
            rasterize_mode=self.config.rasterize_mode,
        )
        strawberry_mask = strawberry_mask[..., 0:1]  # Use the first channel for the mask

        if self.training:
            self.strategy.step_pre_backward(
                self.gauss_params, self.optimizers, self.strategy_state, self.step, self.info
            )
        alpha = alpha[:, ...]

        background = self._get_background_color()
        rgb = render[:, ..., :3] + (1 - alpha) * background
        rgb = torch.clamp(rgb, 0.0, 1.0)

        # apply bilateral grid
        if self.config.use_bilateral_grid and self.training:
            if camera.metadata is not None and "cam_idx" in camera.metadata:
                rgb = self._apply_bilateral_grid(rgb, camera.metadata["cam_idx"], H, W)

        if render_mode == "RGB+ED":
            depth_im = render[:, ..., 3:4]
            depth_im = torch.where(alpha > 0, depth_im, depth_im.detach().max()).squeeze(0)
        else:
            depth_im = None

        if background.shape[0] == 3 and not self.training:
            background = background.expand(H, W, 3)

        # Add overlay output for visualizing bruise mask in splat
        bruise_color = torch.tensor([0.5, 0.0, 0.5], device=rgb.device, dtype=rgb.dtype)  # purple
        bruise_overlay = bruise_mask.squeeze(0).expand(-1, -1, 3) * bruise_color  # [H, W, 3]
        alpha_overlay = 0.7
        rgb_with_bruise = alpha_overlay * bruise_overlay + (1 - alpha_overlay) * rgb.squeeze(0)  # [H, W, 3]

        return {
            "rgb": rgb.squeeze(0),  # type: ignore
            "bruise": bruise_mask.squeeze(0),  # type: ignore
            "strawberry": strawberry_mask.squeeze(0),
            "rgb_with_bruise": rgb_with_bruise,  # type: ignores
            "depth": depth_im,  # type: ignore
            "accumulation": alpha.squeeze(0),  # type: ignore
            "background": background,  # type: ignore
        }  # type: ignore    
    
    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        # Compute loss for bruise parameter
        pred_bruise = outputs["bruise"]

        if "bruise_mask" in batch:
            # batch["bruise_mask"] : [H, W, 1]        
            bruise_mask = downscale_mask_to_pred_shape(batch["bruise_mask"], pred_bruise).to(self.device)

            # compute binary cross entropy loss
            bruise_mask_loss = F.binary_cross_entropy_with_logits(pred_bruise, bruise_mask)
            
        else:
            bruise_mask_loss = torch.tensor(0.0).to(self.device)

        # Compute loss for strawberry parameter
        pred_strawberry = outputs["strawberry"]

        if "strawberry_mask" in batch:
            # batch["strawberry_mask"] : [H, W, 1]        
            strawberry_mask = downscale_mask_to_pred_shape(batch["strawberry_mask"], pred_strawberry).to(self.device)

            # compute binary cross entropy loss
            strawberry_mask_loss = F.binary_cross_entropy_with_logits(pred_strawberry, strawberry_mask)
            
        else:
            strawberry_mask_loss = torch.tensor(0.0).to(self.device)
        
        # Compute loss for RGB image
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        # Create loss dictionary
        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * simloss + bruise_mask_loss + strawberry_mask_loss,
            "scale_reg": scale_reg,
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)
            if self.config.use_bilateral_grid:
                loss_dict["tv_loss"] = 10 * total_variation_loss(self.bil_grids.grids)

        return loss_dict
    
def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def num_sh_bases(degree: int) -> int:
    """
    Returns the number of spherical harmonic bases for a given degree.
    """
    assert degree <= 4, "We don't support degree greater than 4."
    return (degree + 1) ** 2


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def resize_to(image: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
    """
    Safely resize `image` to shape [target_h, target_w, C] using area downscaling.
    Input can be [H, W, C], [C, H, W], or [1, H, W]. We unify to [B=1, C, H, W],
    call F.interpolate(..., size=(target_h, target_w), mode='area'), and return [target_h, target_w, C].
    """
    # 1) Ensure we have a final dimension 'C'. Commonly, the mask is [1, H, W]. If so, move it to [H, W, 1].
    if image.dim() == 3:
        # possible shapes to handle: [H, W, C], [C, H, W], [1, H, W], etc.
        if image.shape[0] == 1:
            # shape is [1, H, W], so permute to [H, W, 1]
            image = image.permute(1, 2, 0)  # => [H, W, 1]
        elif image.shape[-1] == 1:
            # shape [H, W, 1] is already good
            pass
        elif image.shape[0] in [1, 3] and image.shape[-1] not in [1, 3]:
            # shape is [C, H, W] => [H, W, C]
            image = image.permute(1, 2, 0)
    else:
        raise ValueError(f"Expected 3D tensor, got shape {image.shape}")

    H, W, C = image.shape
    target_h, target_w = target_hw

    # 2) If it already matches, skip
    if (H, W) == (target_h, target_w):
        return image

    # 3) Expand to [B, C, H, W], use F.interpolate with mode='area'
    image_4d = image.permute(2, 0, 1).unsqueeze(0)  # => [1, C, H, W]
    image_resized = F.interpolate(image_4d, size=(target_h, target_w), mode="area", align_corners=None)

    # 4) Squeeze back to [target_h, target_w, C]
    image_out = image_resized.squeeze(0).permute(1, 2, 0)  # => [H, W, C]
    return image_out

def downscale_mask_to_pred_shape(mask: torch.Tensor, pred_mask: torch.Tensor) -> torch.Tensor:
    """
    Downscale 'mask' so it exactly matches the shape of 'pred_mask'.
    Ensures final shape is pred_mask.shape[:2] + (mask_channels).
    """
    # pred_mask is [H, W, 1], so we want the same H, W
    target_hw = pred_mask.shape[:2]
    return resize_to(mask, target_hw)

def pcd_to_normal(xyz: Tensor):
        hd, wd, _ = xyz.shape
        bottom_point = xyz[..., 2:hd, 1 : wd - 1, :]
        top_point = xyz[..., 0 : hd - 2, 1 : wd - 1, :]
        right_point = xyz[..., 1 : hd - 1, 2:wd, :]
        left_point = xyz[..., 1 : hd - 1, 0 : wd - 2, :]
        left_to_right = right_point - left_point
        bottom_to_top = top_point - bottom_point
        xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
        xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
        xyz_normal = torch.nn.functional.pad(
            xyz_normal.permute(2, 0, 1), (1, 1, 1, 1), mode="constant"
        ).permute(1, 2, 0)
        return xyz_normal

def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> Tensor:
    """Generates camera pixel coordinates [W,H]

    Returns:
        stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
    """

    # img size is (w,h)
    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",  # W = u by H = v
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )  # stored as (x, y) coordinates
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()
    
    return image_coords
    
def normal_from_depth_image(
    depths: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_size: tuple,
    c2w: torch.Tensor,
    device: torch.device,
    smooth: bool = False,
):
    """estimate normals from depth map"""
    if smooth:
        if torch.count_nonzero(depths) > 0:
            print("Input depth map contains 0 elements, skipping smoothing filter")
        else:
            kernel_size = (9, 9)
            depths = torch.from_numpy(
                cv2.GaussianBlur(depths.cpu().numpy(), kernel_size, 0)
            ).to(device)
    means3d, _ = get_means3d_backproj(depths, fx, fy, int(cx), int(cy), img_size, c2w, device)
    means3d = means3d.view(img_size[1], img_size[0], 3)
    normals = pcd_to_normal(means3d)
    return normals

def get_means3d_backproj(
        depths: torch.Tensor,
        fx: float,
        fy: float,
        cx: int,
        cy: int,
        img_size: tuple,
        c2w: torch.Tensor,
        device: torch.device,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List]:
        """Backprojection using camera intrinsics and extrinsics

        image_coords -> (x,y,depth) -> (X, Y, depth)

        Returns:
            Tuple of (means: Tensor, image_coords: Tensor)
        """

        if depths.dim() == 3:
            depths = depths.view(-1, 1)
        elif depths.shape[-1] != 1:
            depths = depths.unsqueeze(-1).contiguous()
            depths = depths.view(-1, 1)
        if depths.dtype != torch.float:
            depths = depths.float()
            c2w = c2w.float()
        if c2w.device != device:
            c2w = c2w.to(device)

        image_coords = get_camera_coords(img_size)
        image_coords = image_coords.to(device)  # note image_coords is (H,W)
        means3d = torch.empty(size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device).view(-1, 3)
        means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
        means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
        means3d[:, 2] = depths[:, 0]  # z

        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.tensor(mask, device=depths.device)
            means3d = means3d[mask]
            image_coords = image_coords[mask]

        if c2w is None:
            c2w = torch.eye((means3d.shape[0], 4, 4), device=device)

        # to world coords
        means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
        return means3d, image_coords # type: ignore

@torch_compile()
def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

def compute_bruise_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, threshold: float = 0.5) -> float:
        """
        Simple binary IoU for a single image.

        pred_mask:  [H, W] floating bruise, typically in [0,1]
        gt_mask:    [H, W] ground truth bruise, 0 or 1
        threshold:  Pred values > threshold => 1, else 0
        """
        # Binarize predicted bruise:
        pred_binary = (pred_mask > threshold).bool()

        # Ensure GT is also boolean:
        gt_binary = (gt_mask > 0.5).bool()  # or simply (gt_mask == 1) if you know it’s 0/1

        intersection = (pred_binary & gt_binary).sum().float()
        union = (pred_binary | gt_binary).sum().float()

        if union == 0:
            # Edge case: if both are completely empty, define IoU = 1.0 or 0.0 as you prefer
            return 1.0 if intersection == 0 else 0.0
        else:
            return (intersection / union).item()
        
def resize_bruise_mask(bruise_mask: torch.Tensor, shape_des: Tuple[int, int]) -> torch.Tensor:
    """
    Resizes the 'bruise_mask' tensor to match the 'target_shape'.
    Ensures compatibility with tensors having additional dimensions (e.g., [H, W, C] or [1, H, W]).

    Args:
        bruise_mask (torch.Tensor): Input mask tensor. Can be [H, W], [H, W, C], or [1, H, W].
        target_shape (Tuple[int, int]): Target height and width as a tuple.

    Returns:
        torch.Tensor: Resized bruise mask with shape [H, W, C].
    """
    # Ensure 'bruise_mask' has 3 dimensions (e.g., [H, W, C] or [1, H, W])
    if bruise_mask.dim() == 2:  # If [H, W], add a channel dimension
        bruise_mask = bruise_mask.unsqueeze(-1)
    elif bruise_mask.dim() == 3 and bruise_mask.shape[0] == 1:  # If [1, H, W], permute to [H, W, 1]
        bruise_mask = bruise_mask.permute(1, 2, 0)

    # Resize using area interpolation. Convert to [1, C, H, W]. Target (H, W), then convert back to [H, W, C]
    resized = F.interpolate(bruise_mask.permute(2, 0, 1).unsqueeze(0), shape_des, mode="area",).squeeze(0).permute(1, 2, 0)
    return resized

def compute_psnr(pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> float:
    """
    Computes the Peak Signal-to-Noise Ratio (PSNR) between the predicted mask and ground truth.

    Args:
        pred_mask (torch.Tensor): Predicted mask (should be in range [0,1] after sigmoid).
        gt_mask (torch.Tensor): Ground truth mask (binary 0/1).
    
    Returns:
        float: The PSNR value.
    """
    # Ensure masks are in [0,1]
    pred_mask = torch.clamp(pred_mask, 0, 1)
    gt_mask = torch.clamp(gt_mask, 0, 1)

    # Compute Mean Squared Error (MSE)
    mse = F.mse_loss(pred_mask, gt_mask)

    # If MSE is 0, return a high PSNR value
    if mse == 0:
        return float("inf")  # Perfect match case

    # Compute PSNR
    max_val = 1.0  # Since our masks are in [0,1]
    psnr = 10 * torch.log10(max_val**2 / mse)

    return psnr.item()