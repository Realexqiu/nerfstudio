# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors.
# All rights reserved.
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

"""
Script for exporting NeRF into other formats.
"""

from __future__ import annotations

import json
import os
import sys
import typing
from collections import OrderedDict
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast

import numpy as np
import open3d as o3d
import torch
import tyro
from typing_extensions import Annotated, Literal

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager
from nerfstudio.data.datamanagers.parallel_datamanager import ParallelDataManager
from nerfstudio.data.datamanagers.random_cameras_datamanager import RandomCamerasDataManager
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.exporter import texture_utils, tsdf_utils
from nerfstudio.exporter.exporter_utils import collect_camera_poses, generate_point_cloud, get_mesh_from_filename
from nerfstudio.exporter.marching_cubes import generate_mesh_with_multires_marching_cubes
from nerfstudio.fields.sdf_field import SDFField  # noqa
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline, VanillaPipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

# -----------------------------------------------------------------------------------
# A small helper function to apply "manual cube filtering" by:
#   1) computing the mean position in X, Y, Z,
#   2) finding the max distance of any point from that mean along each axis,
#   3) taking the largest of those 3 distances (makes a cube),
#   4) filtering out points not inside that cube.
# We provide an optional `dist_scale` so you can enlarge or shrink the cube boundary.
# -----------------------------------------------------------------------------------
def cube_filter_mask(positions: np.ndarray, dist_scale: float = 1.0) -> np.ndarray:
    """
    Returns a boolean mask selecting positions inside the largest-axis cube
    around the mean. The side extends from (mean - max_dist) to (mean + max_dist)
    in each dimension, where `max_dist` is the maximum of (|X - X_mean|, etc.)
    for all points in X, Y, Z. Then scaled by dist_scale.

    positions: [N, 3]
    dist_scale: scale factor on the half-width of the cube
    """
    x_mean = positions[:, 0].mean()
    y_mean = positions[:, 1].mean()
    z_mean = positions[:, 2].mean()
    print(f"Mean position: ({x_mean:.2f}, {y_mean:.2f}, {z_mean:.2f})")

    x_max_dist = np.max(np.abs(positions[:, 0] - x_mean))
    y_max_dist = np.max(np.abs(positions[:, 1] - y_mean))
    z_max_dist = np.max(np.abs(positions[:, 2] - z_mean))
    max_d = max(x_max_dist, y_max_dist, z_max_dist) * dist_scale
    print("Max distances:", x_max_dist, y_max_dist, z_max_dist)
    print(f"Max distance: {max_d:.2f}")

    mask = (
        (positions[:, 0] >= x_mean - max_d) & (positions[:, 0] <= x_mean + max_d) &
        (positions[:, 1] >= y_mean - max_d) & (positions[:, 1] <= y_mean + max_d) &
        (positions[:, 2] >= z_mean - max_d) & (positions[:, 2] <= z_mean + max_d)
    )
    return mask


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""

    load_config: Path
    """Path to the config YAML file."""
    load_config2: Path
    output_dir: Path
    """Path to the output directory."""


def validate_pipeline(normal_method: str, normal_output_name: str, pipeline: Pipeline) -> None:
    """Check that the pipeline is valid for this exporter.

    Args:
        normal_method: Method to estimate normals with. Either "open3d" or "model_output".
        normal_output_name: Name of the normal output.
        pipeline: Pipeline to evaluate with.
    """
    if normal_method == "model_output":
        CONSOLE.print("Checking that the pipeline has a normal output.")
        origins = torch.zeros((1, 3), device=pipeline.device)
        directions = torch.ones_like(origins)
        pixel_area = torch.ones_like(origins[..., :1])
        camera_indices = torch.zeros_like(origins[..., :1])
        metadata = {"directions_norm": torch.linalg.vector_norm(directions, dim=-1, keepdim=True)}
        ray_bundle = RayBundle(
            origins=origins,
            directions=directions,
            pixel_area=pixel_area,
            camera_indices=camera_indices,
            metadata=metadata,
        )
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print(
                "[bold yellow]Warning: Please train a model with normals "
                "(e.g., nerfacto with predicted normals turned on)."
            )
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class ExportPointCloud(Exporter):
    """Export NeRF as a point cloud."""

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""

    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""
    save_world_frame: bool = False
    """If set, saves the point cloud in the same frame as the original dataset. Otherwise, uses the
    scaled and reoriented coordinate space expected by the NeRF models."""

    # -------- NEW: option to apply manual cube filter to the final point cloud.
    use_cube_filter: bool = False
    """If True, filter the final point cloud to a largest-axis cube around its mean position."""
    cube_dist_scale: float = 0.5
    """Scale factor for the half-width of the bounding cube, if use_cube_filter is True."""

    def main(self) -> None:
        """Export point cloud."""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager, RandomCamerasDataManager),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        crop_obb = None
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)

        # Generate the initial pcd from the pipeline
        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            bruise_output_name="bruise",
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )

        # Optionally transform to the original dataset world frame
        if self.save_world_frame:
            points = np.asarray(pcd.points)
            poses = np.eye(4, dtype=np.float32)[None, ...].repeat(points.shape[0], axis=0)[:, :3, :]
            poses[:, :3, 3] = points
            poses = pipeline.datamanager.train_dataparser_outputs.transform_poses_to_original_space(
                torch.from_numpy(poses)
            )
            points = poses[:, :3, 3].numpy()
            pcd.points = o3d.utility.Vector3dVector(points)

        torch.cuda.empty_cache()

        # -------------------- NEW: Manual Cube Filter in pointcloud. --------------------
        if self.use_cube_filter:
            CONSOLE.print("[bold blue]Applying manual cube filter to point cloud...")
            points_np = np.asarray(pcd.points)  # shape [N, 3]
            mask = cube_filter_mask(points_np, dist_scale=self.cube_dist_scale)
            indices = np.where(mask)[0]
            # update the pcd to keep only those points
            filtered_pcd = pcd.select_by_index(indices)
            pcd = filtered_pcd
            CONSOLE.print(f"[bold blue]Cube filter kept {len(indices)}/{len(points_np)} points.")

        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")
        CONSOLE.print("Saving Point Cloud...")

        # Convert to Open3D Tensor PCD for saving
        tpcd = o3d.t.geometry.PointCloud.from_legacy(pcd)
        # The legacy PLY writer converts colors to UInt8,
        # let us do the same to save space.
        tpcd.point.colors = (tpcd.point.colors * 255).to(o3d.core.Dtype.UInt8)  # type: ignore
        o3d.t.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), tpcd)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")


@dataclass
class ExportTSDFMesh(Exporter):
    """
    Export a mesh using TSDF processing.
    """

    downscale_factor: int = 2
    """Downscale the images starting from the resolution used for training."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    resolution: Union[int, List[int]] = field(default_factory=lambda: [128, 128, 128])
    """Resolution of the TSDF volume or [x, y, z] resolutions individually."""
    batch_size: int = 10
    """How many depth images to integrate per batch."""
    use_bounding_box: bool = True
    """Whether to use a bounding box for the TSDF volume."""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""
    texture_method: Literal["tsdf", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'tsdf' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    refine_mesh_using_initial_aabb_estimate: bool = False
    """Refine the mesh using the initial AABB estimate."""
    refinement_epsilon: float = 1e-2
    """Refinement epsilon for the mesh. This is the distance in meters that the refined AABB/OBB will be expanded by
    in each direction."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        tsdf_utils.export_tsdf_mesh(
            pipeline,
            self.output_dir,
            self.downscale_factor,
            self.depth_output_name,
            self.rgb_output_name,
            self.resolution,
            self.batch_size,
            use_bounding_box=self.use_bounding_box,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            refine_mesh_using_initial_aabb_estimate=self.refine_mesh_using_initial_aabb_estimate,
            refinement_epsilon=self.refinement_epsilon,
        )

        # possibly
        # texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the tsdf export
            mesh = get_mesh_from_filename(
                str(self.output_dir / "tsdf_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportPoissonMesh(Exporter):
    """
    Export a mesh using poisson surface reconstruction.
    """

    num_points: int = 1000000
    """Number of points to generate. May result in less if outlier removal is used."""
    remove_outliers: bool = True
    """Remove outliers from the point cloud."""
    reorient_normals: bool = True
    """Reorient point cloud normals based on view direction."""
    depth_output_name: str = "depth"
    """Name of the depth output."""
    rgb_output_name: str = "rgb"
    """Name of the RGB output."""
    normal_method: Literal["open3d", "model_output"] = "model_output"
    """Method to estimate normals with."""
    normal_output_name: str = "normals"
    """Name of the normal output."""
    save_point_cloud: bool = False
    """Whether to save the point cloud."""
    use_bounding_box: bool = True
    """Only query points within the bounding box"""
    bounding_box_min: Tuple[float, float, float] = (-1, -1, -1)
    """Minimum of the bounding box, used if use_bounding_box is True."""
    bounding_box_max: Tuple[float, float, float] = (1, 1, 1)
    """Maximum of the bounding box, used if use_bounding_box is True."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    num_rays_per_batch: int = 32768
    """Number of rays to evaluate per batch. Decrease if you run out of memory."""
    texture_method: Literal["point_cloud", "nerf"] = "nerf"
    """Method to texture the mesh with. Either 'point_cloud' or 'nerf'."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""
    std_ratio: float = 10.0
    """Threshold based on STD of the average distances across the point cloud to remove outliers."""

    def main(self) -> None:
        """Export mesh"""

        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        validate_pipeline(self.normal_method, self.normal_output_name, pipeline)

        # Increase the batchsize to speed up the evaluation.
        assert isinstance(
            pipeline.datamanager,
            (VanillaDataManager, ParallelDataManager, FullImageDatamanager, RandomCamerasDataManager),
        )
        assert pipeline.datamanager.train_pixel_sampler is not None
        pipeline.datamanager.train_pixel_sampler.num_rays_per_batch = self.num_rays_per_batch

        # Whether the normals should be estimated based on the point cloud.
        estimate_normals = self.normal_method == "open3d"
        if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
            crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
        else:
            crop_obb = None

        pcd = generate_point_cloud(
            pipeline=pipeline,
            num_points=self.num_points,
            remove_outliers=self.remove_outliers,
            reorient_normals=self.reorient_normals,
            estimate_normals=estimate_normals,
            rgb_output_name=self.rgb_output_name,
            depth_output_name=self.depth_output_name,
            normal_output_name=self.normal_output_name if self.normal_method == "model_output" else None,
            crop_obb=crop_obb,
            std_ratio=self.std_ratio,
        )
        torch.cuda.empty_cache()
        CONSOLE.print(f"[bold green]:white_check_mark: Generated {pcd}")

        if self.save_point_cloud:
            CONSOLE.print("Saving Point Cloud...")
            o3d.io.write_point_cloud(str(self.output_dir / "point_cloud.ply"), pcd)
            print("\033[A\033[A")
            CONSOLE.print("[bold green]:white_check_mark: Saving Point Cloud")

        CONSOLE.print("Computing Mesh... this may take a while.")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Computing Mesh")

        CONSOLE.print("Saving Mesh...")
        o3d.io.write_triangle_mesh(str(self.output_dir / "poisson_mesh.ply"), mesh)
        print("\033[A\033[A")
        CONSOLE.print("[bold green]:white_check_mark: Saving Mesh")

        # This will texture the mesh with NeRF and export to a mesh.obj file
        # and a material and texture file
        if self.texture_method == "nerf":
            # load the mesh from the poisson reconstruction
            mesh = get_mesh_from_filename(
                str(self.output_dir / "poisson_mesh.ply"), target_num_faces=self.target_num_faces
            )
            CONSOLE.print("Texturing mesh with NeRF")
            texture_utils.export_textured_mesh(
                mesh,
                pipeline,
                self.output_dir,
                px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
                unwrap_method=self.unwrap_method,
                num_pixels_per_side=self.num_pixels_per_side,
            )


@dataclass
class ExportMarchingCubesMesh(Exporter):
    """Export a mesh using marching cubes."""

    isosurface_threshold: float = 0.0
    """The isosurface threshold for extraction. For SDF based methods the surface is the zero level set."""
    resolution: int = 1024
    """Marching cube resolution."""
    simplify_mesh: bool = False
    """Whether to simplify the mesh."""
    bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum of the bounding box."""
    bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum of the bounding box."""
    px_per_uv_triangle: int = 4
    """Number of pixels per UV triangle."""
    unwrap_method: Literal["xatlas", "custom"] = "xatlas"
    """The method to use for unwrapping the mesh."""
    num_pixels_per_side: int = 2048
    """If using xatlas for unwrapping, the pixels per side of the texture image."""
    target_num_faces: Optional[int] = 50000
    """Target number of faces for the mesh to texture."""

    def main(self) -> None:
        """Main function."""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)

        # This only works for models that have an SDF field. e.g. nerfacto with SDF
        assert hasattr(pipeline.model.config, "sdf_field"), "Model must have an SDF field."

        CONSOLE.print("Extracting mesh with marching cubes... which may take a while")

        assert self.resolution % 512 == 0, f"""resolution must be divisible by 512, got {self.resolution}.
        This is important because the algorithm uses a multi-resolution approach
        to evaluate the SDF where the minimum resolution is 512."""

        # Extract mesh using marching cubes for sdf at a multi-scale resolution.
        multi_res_mesh = generate_mesh_with_multires_marching_cubes(
            geometry_callable_field=lambda x: cast(SDFField, pipeline.model.field)
            .forward_geonetwork(x)[:, 0]
            .contiguous(),
            resolution=self.resolution,
            bounding_box_min=self.bounding_box_min,
            bounding_box_max=self.bounding_box_max,
            isosurface_threshold=self.isosurface_threshold,
            coarse_mask=None,
        )
        filename = self.output_dir / "sdf_marching_cubes_mesh.ply"
        multi_res_mesh.export(filename)

        # load the mesh from the marching cubes export
        mesh = get_mesh_from_filename(str(filename), target_num_faces=self.target_num_faces)
        CONSOLE.print("Texturing mesh with NeRF...")
        texture_utils.export_textured_mesh(
            mesh,
            pipeline,
            self.output_dir,
            px_per_uv_triangle=self.px_per_uv_triangle if self.unwrap_method == "custom" else None,
            unwrap_method=self.unwrap_method,
            num_pixels_per_side=self.num_pixels_per_side,
        )


@dataclass
class ExportCameraPoses(Exporter):
    """
    Export camera poses to a .json file.
    """

    def main(self) -> None:
        """Export camera poses"""
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config)
        assert isinstance(pipeline, VanillaPipeline)
        train_frames, eval_frames = collect_camera_poses(pipeline)

        for file_name, frames in [("transforms_train.json", train_frames), ("transforms_eval.json", eval_frames)]:
            if len(frames) == 0:
                CONSOLE.print(f"[bold yellow]No frames found for {file_name}. Skipping.")
                continue

            output_file_path = os.path.join(self.output_dir, file_name)

            with open(output_file_path, "w", encoding="UTF-8") as f:
                json.dump(frames, f, indent=4)

            CONSOLE.print(f"[bold green]:white_check_mark: Saved poses to {output_file_path}")


# @dataclass
# class ExportGaussianSplat(Exporter):
#     """
#     Export 3D Gaussian Splatting model to a .ply
#     """

#     output_filename: str
#     """Name of the output file."""
#     obb_center: Optional[Tuple[float, float, float]] = None
#     """Center of the oriented bounding box."""
#     obb_rotation: Optional[Tuple[float, float, float]] = None
#     """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
#     obb_scale: Optional[Tuple[float, float, float]] = None
#     """Scale of the oriented bounding box along each axis."""
#     ply_color_mode: Literal["sh_coeffs", "rgb"] = "rgb"
#     """If "rgb", export colors as red/green/blue fields. Otherwise, export colors as
#     spherical harmonics coefficients."""

#     # ---- NEW: manual cube filter for Gaussians.
#     use_cube_filter: bool = True
#     """If True, filter Gaussians so only those in a largest-axis cube around the mean remain."""
#     cube_dist_scale: float = 0.08
#     """Scale factor for half-width of the bounding cube if use_cube_filter is True."""

#     @staticmethod
#     def write_ply(
#         filename: str,
#         count: int,
#         map_to_tensors: typing.OrderedDict[str, np.ndarray],
#     ):
#         """
#         Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
#         Note: All float values will be converted to float32 for writing.

#         Parameters:
#         filename (str): The name of the file to write.
#         count (int): The number of vertices to write.
#         map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays.
#         """
#         if not all(tensor.size == count for tensor in map_to_tensors.values()):
#             raise ValueError("Count does not match the length of all tensors")

#         if not all(
#             isinstance(tensor, np.ndarray)
#             and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
#             and tensor.size > 0
#             for tensor in map_to_tensors.values()
#         ):
#             raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

#         with open(filename, "wb") as ply_file:
#             nerfstudio_version = version("nerfstudio")
#             ply_file.write(b"ply\n")
#             ply_file.write(b"format binary_little_endian 1.0\n")
#             ply_file.write(f"element vertex {count}\n".encode())

#             # Write the properties
#             for key, tensor in map_to_tensors.items():
#                 data_type = "float" if tensor.dtype.kind == "f" else "uchar"
#                 ply_file.write(f"property {data_type} {key}\n".encode())

#             ply_file.write(b"end_header\n")

#             # Write binary data, row by row
#             for i in range(count):
#                 for tensor in map_to_tensors.values():
#                     value = tensor[i]
#                     if tensor.dtype.kind == "f":
#                         ply_file.write(np.float32(value).tobytes())
#                     elif tensor.dtype == np.uint8:
#                         ply_file.write(value.tobytes())

#         CONSOLE.print(f"Point cloud saved to: {filename}")

#     def main(self) -> None:
#         if not self.output_dir.exists():
#             self.output_dir.mkdir(parents=True)

#         _, pipeline, _, _ = eval_setup(self.load_config, test_mode="inference")
#         assert isinstance(pipeline.model, SplatfactoModel)

#         model: SplatfactoModel = pipeline.model
#         filename = self.output_dir / self.output_filename
#         map_to_tensors = OrderedDict()

#         with torch.no_grad():
#             positions = model.means.cpu().numpy()
#             n = positions.shape[0]

#             # Prepare the output dictionary
#             map_to_tensors["x"] = positions[:, 0]
#             map_to_tensors["y"] = positions[:, 1]
#             map_to_tensors["z"] = positions[:, 2]
#             map_to_tensors["nx"] = np.zeros(n, dtype=np.float32)
#             map_to_tensors["ny"] = np.zeros(n, dtype=np.float32)
#             map_to_tensors["nz"] = np.zeros(n, dtype=np.float32)

#             if self.ply_color_mode == "rgb":
#                 colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
#                 colors = (colors * 255).astype(np.uint8)
#                 map_to_tensors["red"] = colors[:, 0]
#                 map_to_tensors["green"] = colors[:, 1]
#                 map_to_tensors["blue"] = colors[:, 2]
#             elif self.ply_color_mode == "sh_coeffs":
#                 shs_0 = model.shs_0.contiguous().cpu().numpy()  # [N, channel]
#                 for i in range(shs_0.shape[1]):
#                     map_to_tensors[f"f_dc_{i}"] = shs_0[:, i, None]

#             if model.config.sh_degree > 0:
#                 if self.ply_color_mode == "rgb":
#                     CONSOLE.print(
#                         "Warning: model has higher-level spherical harmonics, ignoring them and only export rgb."
#                     )
#                 elif self.ply_color_mode == "sh_coeffs":
#                     shs_rest = model.shs_rest.transpose(1, 2).contiguous().cpu().numpy()
#                     shs_rest = shs_rest.reshape((n, -1))
#                     for i in range(shs_rest.shape[-1]):
#                         map_to_tensors[f"f_rest_{i}"] = shs_rest[:, i, None]

#             map_to_tensors["opacity"] = model.opacities.data.cpu().numpy()
#             scales = model.scales.data.cpu().numpy()
#             for i in range(3):
#                 map_to_tensors[f"scale_{i}"] = scales[:, i, None]

#             quats = model.quats.data.cpu().numpy()
#             for i in range(4):
#                 map_to_tensors[f"rot_{i}"] = quats[:, i, None]

#             # Existing oriented bounding box filter
#             if self.obb_center is not None and self.obb_rotation is not None and self.obb_scale is not None:
#                 crop_obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
#                 mask = crop_obb.within(torch.from_numpy(positions)).numpy()
#                 for k, t in map_to_tensors.items():
#                     map_to_tensors[k] = t[mask]
#                 n = map_to_tensors["x"].shape[0]

#             # -------------------- NEW: Manual cube filter for Gaussians. --------------------
#             if self.use_cube_filter:
#                 CONSOLE.print("[bold blue]Applying manual cube filter to Gaussians...")
#                 # Recompute the current positions, since we might have changed them above
#                 px = map_to_tensors["x"]
#                 py = map_to_tensors["y"]
#                 pz = map_to_tensors["z"]
#                 new_positions = np.stack([px, py, pz], axis=-1)  # [n, 3]

#                 mask_cube = cube_filter_mask(new_positions, dist_scale=self.cube_dist_scale)
#                 for k, t in map_to_tensors.items():
#                     map_to_tensors[k] = t[mask_cube]
#                 kept = mask_cube.sum()
#                 CONSOLE.print(f"[bold blue]Cube filter kept {kept}/{len(mask_cube)} gaussians.")
#                 n = kept

#         # Now we do the final NaN / Inf / low opacity checks
#         select = np.ones(n, dtype=bool)
#         for k, t in map_to_tensors.items():
#             n_before = np.sum(select)
#             # ensure finite
#             select = np.logical_and(select, np.isfinite(t).all(axis=-1))
#             n_after = np.sum(select)
#             if n_after < n_before:
#                 diff = n_before - n_after
#                 CONSOLE.print(f"{diff} Gaussians with NaN/Inf in {k}")
#         # Low opacity filter (logit(1/255))
#         low_opacity_gaussians = map_to_tensors["opacity"].squeeze(axis=-1) < -5.5373
#         opa_count = np.sum(low_opacity_gaussians)
#         select[low_opacity_gaussians] = 0

#         final_count = np.sum(select)
#         if final_count < n:
#             CONSOLE.print(
#                 f"Filtering removed {n - final_count} gaussians total "
#                 f"(NaN/Inf or low opacity). Now exporting {final_count} of {n}."
#             )
#             for k, t in map_to_tensors.items():
#                 map_to_tensors[k] = t[select]

#         # Write out to PLY
#         ExportGaussianSplat.write_ply(str(filename), final_count, map_to_tensors)

@dataclass
class ExportGaussianSplat(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """

    output_filename: str
    obb_center: Optional[Tuple[float, float, float]] = None
    obb_rotation: Optional[Tuple[float, float, float]] = None
    obb_scale: Optional[Tuple[float, float, float]] = None
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "rgb"

    use_cube_filter: bool = True
    cube_dist_scale: float = 0.09

    @staticmethod
    def write_ply(filename: str, count: int, t_map: typing.OrderedDict[str, np.ndarray]):
        if not all(t.size == count for t in t_map.values()):
            raise ValueError("Count does not match the length of all tensors")

        if not all(isinstance(t, np.ndarray) and (t.dtype.kind == "f" or t.dtype == np.uint8) and t.size > 0 for t in t_map.values()):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            nerfstudio_version = version("nerfstudio")
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")
            ply_file.write(f"element vertex {count}\n".encode())

            for key, tensor in t_map.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            for i in range(count):
                for tensor in t_map.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

        CONSOLE.print(f"Point cloud saved to: {filename}")

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        # Load the first splat
        _, pipeline1, _, _ = eval_setup(self.load_config, test_mode="inference")
        assert isinstance(pipeline1.model, SplatfactoModel)
        model1: SplatfactoModel = pipeline1.model

        # Load the second splat
        _, pipeline2, _, _ = eval_setup(self.load_config2, test_mode="inference")
        assert isinstance(pipeline2.model, SplatfactoModel)
        model2: SplatfactoModel = pipeline2.model

        filename = self.output_dir / self.output_filename

        merging_transform_matrix = np.array([
            [ 0.9962653,   0.08359022, -0.02163598,  0.09713192],
            [-0.08630886,  0.97131675, -0.22157292,  0.09418107],
            [ 0.00249406,  0.22261278,  0.97490376, -0.22236903],
            [ 0.0,         0.0,         0.0,         1.0]
        ])

        # 1) Gather data from both splats
        with torch.no_grad():
            # --- Splat from first config ---
            positions1 = model1.means.cpu().numpy()
            n1 = positions1.shape[0]
            map1 = OrderedDict()
            map1["x"] = positions1[:, 0]
            map1["y"] = positions1[:, 1]
            map1["z"] = positions1[:, 2]
            map1["nx"] = np.zeros(n1, dtype=np.float32)
            map1["ny"] = np.zeros(n1, dtype=np.float32)
            map1["nz"] = np.zeros(n1, dtype=np.float32)

            colors1 = torch.clamp(model1.colors.clone(), 0.0, 1.0).data.cpu().numpy()
            colors1 = (colors1 * 255).astype(np.uint8)
            map1["red"]   = colors1[:, 0]
            map1["green"] = colors1[:, 1]
            map1["blue"]  = colors1[:, 2]
            map1["opacity"] = model1.opacities.data.cpu().numpy()
            scales1 = model1.scales.data.cpu().numpy()  # [N,3]
            for i in range(3):
                map1[f"scale_{i}"] = scales1[:, i, None]
            quats1 = model1.quats.data.cpu().numpy()  # [N,4]
            for i in range(4):
                map1[f"rot_{i}"] = quats1[:, i, None]

            # --- Splat from second config ---
            positions2 = model2.means.cpu().numpy()
            n2 = positions2.shape[0]
            map2 = OrderedDict()
            map2["x"] = positions2[:, 0]
            map2["y"] = positions2[:, 1]
            map2["z"] = positions2[:, 2]
            map2["nx"] = np.zeros(n2, dtype=np.float32)
            map2["ny"] = np.zeros(n2, dtype=np.float32)
            map2["nz"] = np.zeros(n2, dtype=np.float32)
            colors2 = torch.clamp(model2.colors.clone(), 0.0, 1.0).data.cpu().numpy()
            colors2 = (colors2 * 255).astype(np.uint8)
            map2["red"]   = colors2[:, 0]
            map2["green"] = colors2[:, 1]
            map2["blue"]  = colors2[:, 2]
            map2["opacity"] = model2.opacities.data.cpu().numpy()
            scales2 = model2.scales.data.cpu().numpy()  # [N,3]
            for i in range(3):
                map2[f"scale_{i}"] = scales2[:, i, None]
            quats2 = model2.quats.data.cpu().numpy()  # [N,4]
            for i in range(4):
                map2[f"rot_{i}"] = quats2[:, i, None]

            ### ADDED CODE: Define two crop boxes (min and max bounds) for each splat
            # block (from world folder)
            crop1_min = np.array([-0.32, -0.29, -0.23])
            crop1_max = np.array([ 0.34,  0.3,   0.025])
            # peg (from world folder)
            crop2_min = np.array([-0.25, -0.32, -0.23])
            crop2_max = np.array([ 0.34,  0.3,   0.025])
            
            # Define transformation matrices for the crop boxes
            # T_crop1 = np.array([
            #     [ 0.97817176,  0.17294787,  0.11519134, -0.11617147],
            #     [ 0.17294787, -0.37028751, -0.91267529, -0.0540201 ],
            #     [-0.11519134,  0.91267529, -0.39211575, -0.97243527],
            #     [ 0.0,         0.0,         0.0,         1.0]
            # ])

            # T_crop2 = np.array([
            #     [ 0.92082825,  0.34466638,  0.18242919,  0.00276074],
            #     [ 0.34466638, -0.50047108, -0.7941875,   0.12648789],
            #     [-0.18242919,  0.7941875,  -0.57964283, -0.81000036],
            #     [ 0.0,         0.0,         0.0,         1.0]
            # ])

            T_crop1 = np.array([
                [ 0.97817176,  0.17294787,  0.11519134, -0.11617147],
                [ 0.17294787, -0.37028751, -0.91267529, -0.0540201 ],
                [-0.11519134,  0.91267529, -0.39211575, -0.97243527],
                [ 0.0,         0.0,         0.0,         1.0]
            ])

            T_crop2 = np.array([
                [ 0.92082825,  0.34466638,  0.18242919,  0.00276074],
                [ 0.34466638, -0.50047108, -0.7941875,   0.12648789],
                [-0.18242919,  0.7941875,  -0.57964283, -0.81000036],
                [ 0.0,         0.0,         0.0,         1.0]
            ])

            

            # ===== ADDED EXTRA ROTATION FOR CROP BOXES =====
            # These parameters allow you to manually adjust the crop box orientation.
            crop1_rotation_deg = 25.0   # adjust this value for splat1 as needed
            crop2_rotation_deg = 25.0   # adjust this value for splat2 as needed

            R_crop1_extra = rotation_matrix_axis_angle(np.array([0, 0, 1], dtype=np.float64), np.radians(crop1_rotation_deg))
            R_crop2_extra = rotation_matrix_axis_angle(np.array([0, 0, 1], dtype=np.float64), np.radians(crop2_rotation_deg))

            T_extra_crop1 = np.eye(4)
            T_extra_crop1[:3, :3] = R_crop1_extra

            T_extra_crop2 = np.eye(4)
            T_extra_crop2[:3, :3] = R_crop2_extra

            # Modify the inverse crop transforms by including these extra rotations:
            T_crop1_modified = T_extra_crop1 @ T_crop1
            T_crop2_modified = T_extra_crop2 @ T_crop2
            # ===== END EXTRA ROTATION =====

            T_crop1_inv = np.linalg.inv(T_crop1_modified)
            T_crop2_inv = np.linalg.inv(T_crop2_modified)
            
            # Apply cropping to splat 1 (map1) using crop1 bounds, using inverse of T_crop1
            new_crop1_min, new_crop1_max = transform_box_crop(crop1_min, crop1_max, T_crop1_inv)
            mask1 = np.logical_and.reduce([
                map1["x"] >= new_crop1_min[0],
                map1["x"] <= new_crop1_max[0],
                map1["y"] >= new_crop1_min[1],
                map1["y"] <= new_crop1_max[1],
                map1["z"] >= new_crop1_min[2],
                map1["z"] <= new_crop1_max[2]
            ])
            for k, t in map1.items():
                map1[k] = t[mask1]
            
            # Apply cropping to splat 2 (map2) using crop2 bounds transformed by T_crop2, using inverse of T_crop2
            new_crop2_min, new_crop2_max = transform_box_crop(crop2_min, crop2_max, T_crop2_inv)
            mask2 = np.logical_and.reduce([
                map2["x"] >= new_crop2_min[0],
                map2["x"] <= new_crop2_max[0],
                map2["y"] >= new_crop2_min[1],
                map2["y"] <= new_crop2_max[1],
                map2["z"] >= new_crop2_min[2],
                map2["z"] <= new_crop2_max[2]
            ])
            for k, t in map2.items():
                map2[k] = t[mask2]
            
            ### END ADDED CODE

            # Visualize the cropped point clouds for each splat before proceeding            
            visualize_map(map1, "Cropped Splat 1")
            visualize_map(map2, "Cropped Splat 2")

            # Apply transformation to the second splat if specified 
            if merging_transform_matrix is not None:
                T = merging_transform_matrix  # Expected to be a 4x4 matrix.

                # Transform positions
                pos2 = np.stack([map2["x"], map2["y"], map2["z"]], axis=-1)  # shape (n2, 3) from cropped data
                pos2_hom = np.concatenate([pos2, np.ones((pos2.shape[0], 1))], axis=-1)  # (n2, 4)
                pos2_transformed = (T @ pos2_hom.T).T  # (n2, 4)
                pos2_transformed = pos2_transformed[:, :3]
                map2["x"] = pos2_transformed[:, 0]
                map2["y"] = pos2_transformed[:, 1]
                map2["z"] = pos2_transformed[:, 2]

                # Transform scales.
                # For each axis, multiply by the norm of the corresponding column of the upper-left 3x3 block.
                scale_factors = np.linalg.norm(T[:3, :3], axis=0)  # shape (3,)
                for i in range(3):
                    map2[f"scale_{i}"] = map2[f"scale_{i}"] * scale_factors[i]

                # Transform rotations.
                # Extract the rotation part from T and remove any scaling.
                R = T[:3, :3]
                R_normalized = R / np.linalg.norm(R, axis=0, keepdims=True)
                # Convert the rotation matrix to a quaternion.
                quat_T = matrix_to_quaternion(R_normalized)
                # For each Gaussian, multiply the transformation quaternion with the original quaternion.
                quats = np.stack([map2[f"rot_{i}"].squeeze() for i in range(4)], axis=-1)  # shape (n2, 4)
                new_quats = np.array([quaternion_multiply(quat_T, q) for q in quats])
                for i in range(4):
                    map2[f"rot_{i}"] = new_quats[:, i][:, None]

            # --- Combine the two splats ---
            map_to_tensors = OrderedDict()

            # We assume both map1 and map2 share the same keys.
            for key in map1.keys():
                map_to_tensors[key] = np.concatenate([map1[key], map2[key]], axis=0)

            # Get number of Gaussians
            num_gaussians = map_to_tensors["x"].shape[0]

            # NaN/Inf filtering and low opacity removal
            select = np.ones(num_gaussians, dtype=bool)
            for k, t in map_to_tensors.items():
                data_check = t[:, 0] if (t.ndim == 2 and t.shape[1] == 1) else t
                select = np.logical_and(select, np.isfinite(data_check).all(axis=-1))

            opa_key = map_to_tensors["opacity"].squeeze(axis=-1)
            low_opa_mask = opa_key < -5.5373
            select[low_opa_mask] = False
            final_count = np.sum(select)
            if final_count < num_gaussians:
                diff = num_gaussians - final_count
                CONSOLE.print(f"Filtering removed {diff} gaussians (NaN/Inf or low opacity). Now {final_count} remain.")
                for k in map_to_tensors.keys():
                    map_to_tensors[k] = map_to_tensors[k][select]

        ExportGaussianSplat.write_ply(str(filename), final_count, map_to_tensors)
        CONSOLE.print(f"[bold green]Done. Exported {final_count} points to {filename}")


@dataclass
class ExportGaussianSplatData:
    """
    Export 3D Gaussian Splatting model to a .npz file with all relevant parameters.
    """

    load_config: Path
    output_dir: Path
    output_filename: str = "gaussians.npz"

    obb_center: Optional[Tuple[float, float, float]] = None
    obb_rotation: Optional[Tuple[float, float, float]] = None
    obb_scale: Optional[Tuple[float, float, float]] = None

    color_mode: Literal["sh_coeffs", "rgb"] = "rgb"
    use_cube_filter: bool = True
    cube_dist_scale: float = 0.07

    max_gaussians: int = 1_000_000
    """
    We'll randomly downsample if the total gaussians exceed this number.
    """

    def main(self) -> None:
        if not self.output_dir.exists():
            self.output_dir.mkdir(parents=True)

        _, pipeline, _, _ = eval_setup(self.load_config, test_mode="inference")
        assert isinstance(pipeline.model, SplatfactoModel)

        model: SplatfactoModel = pipeline.model

        with torch.no_grad():
            positions = model.means.cpu().numpy()
            scales = model.scales.cpu().numpy()
            quats = model.quats.cpu().numpy()
            opacities = model.opacities.cpu().numpy()
            opacities = np.squeeze(opacities, axis=-1)
            colors = torch.clamp(model.colors, 0.0, 1.0).cpu().numpy()

            N = positions.shape[0]
            CONSOLE.print(f"[bold green]Total Gaussians in model: {N}")
            print("positions.shape =", positions.shape)
            print("scales.shape =", scales.shape)
            print("quats.shape =", quats.shape)
            print("opacities.shape =", opacities.shape)

            # Step 1: random downsample if needed
            if N > self.max_gaussians:
                CONSOLE.print(f"[yellow]Found {N} gaussians, downsampling to {self.max_gaussians}.")
                idx = np.random.choice(N, self.max_gaussians, replace=False)
                positions = positions[idx]
                scales = model.scales.cpu().numpy()[idx]
                quats = model.quats.cpu().numpy()[idx]
                opacities = model.opacities.cpu().numpy()[idx]
                colors = torch.clamp(model.colors, 0.0, 1.0).cpu().numpy()[idx]

            else:
                scales = model.scales.cpu().numpy()
                quats = model.quats.cpu().numpy()
                opacities = model.opacities.cpu().numpy()
                colors = torch.clamp(model.colors, 0.0, 1.0).cpu().numpy()

            # Step 3: manual cube filter
            if self.use_cube_filter:
                mask_cube = cube_filter_mask(positions, dist_scale=self.cube_dist_scale)
            else:
                mask_cube = np.ones(positions.shape[0], dtype=bool)

            # # Step 4: finite checks
            # finite_mask = np.isfinite(positions).all(axis=-1) & np.isfinite(scales).all(axis=-1)
            # finite_mask = finite_mask & np.isfinite(quats).all(axis=-1) & np.isfinite(opacities)
            # import pdb; pdb.set_trace()

            final_mask = mask_cube

            positions = positions[final_mask]
            scales = scales[final_mask]
            quats = quats[final_mask]
            opacities = opacities[final_mask]
            colors = colors[final_mask]

            # Step 5: save dict
            data_dict = OrderedDict()
            data_dict["positions"] = positions
            data_dict["scales"] = scales
            data_dict["quaternions"] = quats
            data_dict["opacities"] = opacities

            if self.color_mode == "rgb" and colors is not None:
                data_dict["rgbs"] = colors

        outfile = self.output_dir / self.output_filename
        np.savez(outfile, **data_dict)
        CONSOLE.print(f"[green]Exported {positions.shape[0]} gaussians to {outfile}")

# ----------------------------------------------------------------------------------------
# Combine all exporters into subcommands:
# ----------------------------------------------------------------------------------------
Commands = tyro.conf.FlagConversionOff[
    Union[
        Annotated[ExportPointCloud, tyro.conf.subcommand(name="pointcloud")],
        Annotated[ExportTSDFMesh, tyro.conf.subcommand(name="tsdf")],
        Annotated[ExportPoissonMesh, tyro.conf.subcommand(name="poisson")],
        Annotated[ExportMarchingCubesMesh, tyro.conf.subcommand(name="marching-cubes")],
        Annotated[ExportCameraPoses, tyro.conf.subcommand(name="cameras")],
        Annotated[ExportGaussianSplat, tyro.conf.subcommand(name="gaussian-splat")],
        Annotated[ExportGaussianSplatData, tyro.conf.subcommand(name="gaussian-splat-data")],
    ]
]


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(Commands).main()


if __name__ == "__main__":
    entrypoint()


def get_parser_fn():
    """Get the parser function for the sphinx docs."""
    return tyro.extras.get_parser(Commands)  # noqa

def matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3x3 rotation matrix to a quaternion in (x, y, z, w) order.
    """
    q = np.empty((4,))
    trace = np.trace(R)
    if trace > 0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q[3] = 0.25 * s
        q[0] = (R[2, 1] - R[1, 2]) / s
        q[1] = (R[0, 2] - R[2, 0]) / s
        q[2] = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            q[3] = (R[2, 1] - R[1, 2]) / s
            q[0] = 0.25 * s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            q[3] = (R[0, 2] - R[2, 0]) / s
            q[0] = (R[0, 1] + R[1, 0]) / s
            q[1] = 0.25 * s
            q[2] = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            q[3] = (R[1, 0] - R[0, 1]) / s
            q[0] = (R[0, 2] + R[2, 0]) / s
            q[1] = (R[1, 2] + R[2, 1]) / s
            q[2] = 0.25 * s
    return q

def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Multiply two quaternions (x, y, z, w).
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return np.array([x, y, z, w])

def transform_box_crop(min_bound, max_bound, T):
    corners = np.array([
        [min_bound[0], min_bound[1], min_bound[2]],
        [min_bound[0], min_bound[1], max_bound[2]],
        [min_bound[0], max_bound[1], min_bound[2]],
        [min_bound[0], max_bound[1], max_bound[2]],
        [max_bound[0], min_bound[1], min_bound[2]],
        [max_bound[0], min_bound[1], max_bound[2]],
        [max_bound[0], max_bound[1], min_bound[2]],
        [max_bound[0], max_bound[1], max_bound[2]]
    ])
    corners_hom = np.concatenate([corners, np.ones((8, 1))], axis=1)
    transformed_corners = (T @ corners_hom.T).T[:, :3]
    new_min = transformed_corners.min(axis=0)
    new_max = transformed_corners.max(axis=0)
    return new_min, new_max

def rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Create a 3x3 rotation matrix for rotating 'angle' radians around a normalized 'axis'.
    """
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
         [x*x*C + c,   x*y*C - z*s, x*z*C + y*s],
         [y*x*C + z*s, y*y*C + c,   y*z*C - x*s],
         [z*x*C - y*s, z*y*C + x*s, z*z*C + c]
    ], dtype=np.float64)
    return R

def visualize_map(map_dict, window_name):
    pcd = o3d.geometry.PointCloud()
    pts = np.stack([map_dict["x"], map_dict["y"], map_dict["z"]], axis=-1)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if "red" in map_dict and "green" in map_dict and "blue" in map_dict:
        cols = np.stack([map_dict["red"], map_dict["green"], map_dict["blue"]], axis=-1).astype(np.float64) / 255.0
        pcd.colors = o3d.utility.Vector3dVector(cols)
    o3d.visualization.draw_geometries([pcd], window_name=window_name) # type:ignore