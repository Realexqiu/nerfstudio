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

"""
Script for exporting Bruisefacto into point cloud.
"""

from __future__ import annotations

import sys
import typing
from collections import OrderedDict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import tyro
from typing_extensions import Annotated, Literal
import open3d as o3d

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.bruisefacto.bruisefacto.bruisefacto import BruisefactoModel
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class Exporter:
    """Export the mesh from a YML config to a folder."""
    load_config: Path
    """Path to the config YAML file."""
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
        camera_inds = torch.zeros_like(origins[..., :1])
        metadata = {"directions_norm": torch.linalg.vector_norm(directions, dim=-1, keepdim=True)}
        ray_bundle = RayBundle(origins, directions, pixel_area, camera_inds, metadata)
        outputs = pipeline.model(ray_bundle)
        if normal_output_name not in outputs:
            CONSOLE.print(f"[bold yellow]Warning: Normal output '{normal_output_name}' not found in pipeline outputs.")
            CONSOLE.print(f"Available outputs: {list(outputs.keys())}")
            CONSOLE.print("[bold yellow]Warning: Please train a model with normals (e.g., nerfacto with predicted normals turned on).")
            CONSOLE.print("[bold yellow]Warning: Or change --normal-method")
            CONSOLE.print("[bold yellow]Exiting early.")
            sys.exit(1)


@dataclass
class Export3DGS(Exporter):
    """
    Export 3D Gaussian Splatting model to a .ply
    """
    output_filename: str
    """Name of the output file."""
    obb_center: Optional[Tuple[float, float, float]] = None
    """Center of the oriented bounding box."""
    obb_rotation: Optional[Tuple[float, float, float]] = None
    """Rotation of the oriented bounding box. Expressed as RPY Euler angles in radians"""
    obb_scale: Optional[Tuple[float, float, float]] = None
    """Scale of the oriented bounding box along each axis."""
    ply_color_mode: Literal["sh_coeffs", "rgb"] = "rgb"
    """If "rgb", export colors as red/green/blue fields. Otherwise, export colors as
    spherical harmonics coefficients."""

    bruise_threshold: float = 0.3
    strawberry_threshold: float = 0.5

    @staticmethod
    def write_ply(filename: str, count: int, map_to_tensors: typing.OrderedDict[str, np.ndarray]):
        """
        Writes a PLY file with given vertex properties and a tensor of float or uint8 values in the order specified by the OrderedDict.
        Note: All float values will be converted to float32 for writing.
        Parameters:
        filename (str): The name of the file to write.
        count (int): The number of vertices to write.
        map_to_tensors (OrderedDict[str, np.ndarray]): An ordered dictionary mapping property names to numpy arrays of float or uint8 values.
            Each array should be 1-dimensional and of equal length matching 'count'. Arrays should not be empty.
        """
        # Ensure count matches the length of all tensors
        if not all(tensor.size == count for tensor in map_to_tensors.values()):
            raise ValueError("Count does not match the length of all tensors")

        # Type check for numpy arrays of type float or uint8 and non-empty
        if not all(
            isinstance(tensor, np.ndarray)
            and (tensor.dtype.kind == "f" or tensor.dtype == np.uint8)
            and tensor.size > 0
            for tensor in map_to_tensors.values()
        ):
            raise ValueError("All tensors must be numpy arrays of float or uint8 type and not empty")

        with open(filename, "wb") as ply_file:
            nerfstudio_version = version("nerfstudio")
            # Write PLY header
            ply_file.write(b"ply\n")
            ply_file.write(b"format binary_little_endian 1.0\n")  # Specify format
            ply_file.write(f"element vertex {count}\n".encode())  # Write number of vertices

            # Write properties, in order due to OrderedDict
            for key, tensor in map_to_tensors.items():
                data_type = "float" if tensor.dtype.kind == "f" else "uchar"
                ply_file.write(f"property {data_type} {key}\n".encode())

            ply_file.write(b"end_header\n")

            # Write binary data
            # Note: If this is a performance bottleneck consider using numpy.hstack for efficiency improvement
            for i in range(count):
                for tensor in map_to_tensors.values():
                    value = tensor[i]
                    if tensor.dtype.kind == "f":
                        ply_file.write(np.float32(value).tobytes())
                    elif tensor.dtype == np.uint8:
                        ply_file.write(value.tobytes())

        CONSOLE.print(f"Point cloud saved to: {filename}")

    def main(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _, pipeline, _, _ = eval_setup(self.load_config, test_mode="inference")
        model: BruisefactoModel = pipeline.model
        filename = self.output_dir / self.output_filename

        bruise_threshold = self.bruise_threshold    
        strawberry_threshold = self.strawberry_threshold

        # Gather all Gaussian parameters
        with torch.no_grad():
            # Base positions
            pos = model.means.cpu().numpy()
            mask = None
            if self.obb_center and self.obb_rotation and self.obb_scale:
                obb = OrientedBox.from_params(self.obb_center, self.obb_rotation, self.obb_scale)
                mask = obb.within(torch.from_numpy(pos)).numpy()
                pos = pos[mask]

            map_to_tensors: OrderedDict[str, np.ndarray] = OrderedDict()
            map_to_tensors["x"] = pos[:, 0].astype(np.float32)
            map_to_tensors["y"] = pos[:, 1].astype(np.float32)
            map_to_tensors["z"] = pos[:, 2].astype(np.float32)

            # Colors & opacity
            cols = (torch.clamp(model.colors, 0, 1).cpu().numpy() * 255).astype(np.uint8)
            if mask is not None:
                cols = cols[mask]
            map_to_tensors["red"]   = cols[:, 0]
            map_to_tensors["green"] = cols[:, 1]
            map_to_tensors["blue"]  = cols[:, 2]
            ops = model.opacities.cpu().numpy().astype(np.float32).squeeze(-1)
            if mask is not None:
                ops = ops[mask]
            map_to_tensors["opacity"] = ops

            # Strawberry probabilities
            straw = torch.sigmoid(model.strawberry).cpu().numpy().squeeze(-1)
            if mask is not None:
                straw = straw[mask]
            map_to_tensors["strawberry_prob"] = straw.astype(np.float32)

            # Bruise coloring
            bruise = torch.sigmoid(model.bruise).cpu().numpy().squeeze(-1)
            if mask is not None:
                bruise = bruise[mask]
            bruise_mask = bruise > bruise_threshold
            map_to_tensors["red"][:]   = 255
            map_to_tensors["green"][:] = 0
            map_to_tensors["blue"][:]  = 0
            map_to_tensors["blue"][bruise_mask] = 255

        # Filter out non-finite & low-opacity
        length = map_to_tensors["x"].shape[0]
        valid = np.isfinite(map_to_tensors["x"]) & np.isfinite(map_to_tensors["y"]) & np.isfinite(map_to_tensors["z"])
        valid &= map_to_tensors["opacity"] >= (1/255)
        for k in list(map_to_tensors):
            map_to_tensors[k] = map_to_tensors[k][valid]

        # Build points after validity filter
        pts = np.stack([map_to_tensors["x"], map_to_tensors["y"], map_to_tensors["z"]], axis=1)

        # Box filter
        bounds = {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [-0.5, 0.5]}
        in_box = (
            (pts[:,0] >= bounds["x"][0]) & (pts[:,0] <= bounds["x"][1]) &
            (pts[:,1] >= bounds["y"][0]) & (pts[:,1] <= bounds["y"][1]) &
            (pts[:,2] >= bounds["z"][0]) & (pts[:,2] <= bounds["z"][1])
        )

        # Strawberry threshold
        straw_thresh = map_to_tensors["strawberry_prob"] > strawberry_threshold

        # Final combined mask
        final_mask = in_box & straw_thresh
        for k in list(map_to_tensors):
            map_to_tensors[k] = map_to_tensors[k][final_mask]
        count = int(np.sum(final_mask))

        # Write PLY
        Export3DGS.write_ply(str(filename), count, map_to_tensors)


def create_axes_points(scale=1.0, num_points=40, highlight_axis=None):
    """Create coordinate axes points with optional highlighting of a specific axis.
    
    Args:
        scale: Scale of the axes
        num_points: Number of points per axis
        highlight_axis: Which axis to highlight ('x', 'y', 'z', or None for all)
    """
    # Create points along the X axis from the origin to 'scale'
    x_vals = np.linspace(-1 * scale, scale, num_points)[:, None]
    x_axis = np.hstack([x_vals, np.zeros_like(x_vals), np.zeros_like(x_vals)])
    # Create points along the Y axis
    y_vals = np.linspace(-1 * scale, scale, num_points)[:, None]
    y_axis = np.hstack([np.zeros_like(y_vals), y_vals, np.zeros_like(y_vals)])
    # Create points along the Z axis
    z_vals = np.linspace(-1 * scale, scale, num_points)[:, None]
    z_axis = np.hstack([np.zeros_like(z_vals), np.zeros_like(z_vals), z_vals])
    
    # Concatenate all axis points together
    axes_points = np.concatenate([x_axis, y_axis, z_axis], axis=0)
    
    # Color the axes based on highlight_axis
    default_color = np.array([0.3, 0.3, 0.3])  # Dark gray for non-highlighted axes
    
    if highlight_axis == 'x':
        x_colors = np.tile(np.array([[1, 0, 0]]), (num_points, 1))  # Red for X
        y_colors = np.tile(default_color[None, :], (num_points, 1))
        z_colors = np.tile(default_color[None, :], (num_points, 1))
    elif highlight_axis == 'y':
        x_colors = np.tile(default_color[None, :], (num_points, 1))
        y_colors = np.tile(np.array([[0, 1, 0]]), (num_points, 1))  # Green for Y
        z_colors = np.tile(default_color[None, :], (num_points, 1))
    elif highlight_axis == 'z':
        x_colors = np.tile(default_color[None, :], (num_points, 1))
        y_colors = np.tile(default_color[None, :], (num_points, 1))
        z_colors = np.tile(np.array([[0, 0, 1]]), (num_points, 1))  # Blue for Z
    else:
        # Default: color all axes normally (for backward compatibility)
        x_colors = np.tile(np.array([[1, 0, 0]]), (num_points, 1))
        y_colors = np.tile(np.array([[0, 1, 0]]), (num_points, 1))
        z_colors = np.tile(np.array([[0, 0, 1]]), (num_points, 1))
    
    axes_colors = np.concatenate([x_colors, y_colors, z_colors], axis=0)
    
    axes_pcd = o3d.geometry.PointCloud()
    axes_pcd.points = o3d.utility.Vector3dVector(axes_points)
    axes_pcd.colors = o3d.utility.Vector3dVector(axes_colors)
    
    return axes_pcd


# Commands = tyro.conf.FlagConversionOff[
#     Union[
#         Annotated[Export3DGS, tyro.conf.subcommand(name="bruisefacto")],
#     ]
# ]

# def entrypoint():
#     """Entrypoint for use with pyproject scripts."""
#     tyro.extras.set_accent_color("bright_yellow")
#     tyro.cli(Commands).main()

# if __name__ == "__main__":
#     entrypoint()

# def get_parser_fn():
#     """Get the parser function for the sphinx docs."""
#     return tyro.extras.get_parser(Commands)  # noqa
