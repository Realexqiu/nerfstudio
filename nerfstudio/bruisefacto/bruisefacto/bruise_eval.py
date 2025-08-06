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

#!/usr/bin/env python
"""
bruise_eval.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
import copy
import plotly.graph_objects as go
import pandas as pd
import struct
import open3d as o3d
import torch

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.bruisefacto.bruisefacto.bruisefacto_exporter import Export3DGS
from nerfstudio.bruisefacto.bruisefacto import bruise_eval_utils as util

## To run bruise_eval.py, use the following command: ns-bruise-eval --config-pre /path/to/config_pre.yaml --config-post /path/to/config_post.yaml --output-path /path/to/output.json

@dataclass
class ComputeBruiseMetrics:
    """Load a checkpoint, compute some Bruise metrics, and save it to a JSON file."""

    # Path to config YAML file for pre and post splats. Tyro automatically creates CLI arguments from class attributes
    config_pre: Path
    config_post: Path

    # Output directory where PLY files and JSON will be saved
    output_dir: Path = Path(".")

    # Name of the output .json file (will be saved in output_dir)
    output_filename: str = "bruise_metrics.json"

    bruise_threshold: float = 0.3
    strawberry_threshold: float = 0.5

    @property
    def output_path(self) -> Path:
        """Path where the JSON file will be saved."""
        return self.output_dir / self.output_filename

    # Path to save rendered outputs to.
    render_output_path: Path = output_dir

    def main(self) -> None:
        """Main function."""

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load pre-splat config
        pre_config, pre_pipeline, pre_checkpoint_path, _ = eval_setup(self.config_pre)

        # Load post-splat config
        post_config, post_pipeline, post_checkpoint_path, _ = eval_setup(self.config_post)

        assert self.output_path.suffix == ".json"

        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)

        # # Get the average evaluation image metrics
        # pre_metrics_dict = pre_pipeline.get_average_eval_image_metrics(output_path=self.output_dir, get_std=True)
        # post_metrics_dict = post_pipeline.get_average_eval_image_metrics(output_path=self.output_dir, get_std=True)

        CONSOLE.print(f"[bold cyan] Using Bruise Threshold: {self.bruise_threshold}")
        CONSOLE.print(f"[bold cyan] Using Strawberry Threshold: {self.strawberry_threshold}")

        # Export point clouds for pre and post splats (bruisefacto_exporter.py where you pick points)
        pre_gs_exp = Export3DGS(
            load_config=self.config_pre, 
            output_dir=self.output_dir, 
            output_filename="pre_splat.ply", 
            bruise_threshold=self.bruise_threshold, 
            strawberry_threshold=self.strawberry_threshold
            )
        post_gs_exp = Export3DGS(
            load_config=self.config_post, 
            output_dir=self.output_dir, 
            output_filename="post_splat.ply", 
            bruise_threshold=self.bruise_threshold, 
            strawberry_threshold=self.strawberry_threshold
            )
        pre_gs_exp.main()
        post_gs_exp.main()

        # Define point cloud paths
        pre_splat_path = self.output_dir / "pre_splat.ply"
        post_splat_path = self.output_dir / "post_splat.ply"

        # Read and visualize the point clouds from the output directory
        pre_splat_ply_dataframe = util.load_ply_with_open3d(pre_splat_path)
        post_splat_ply_dataframe = util.load_ply_with_open3d(post_splat_path)
        pre_splat_ply = util.filter_point_cloud_from_dataframe(pre_splat_ply_dataframe)
        post_splat_ply = util.filter_point_cloud_from_dataframe(post_splat_ply_dataframe)
        # util.show_point_cloud(pre_splat_ply)  # Shows point cloud in local server
        # util.show_point_cloud(post_splat_ply)  # Shows point cloud in local server

        # # Align point clouds and calculate max chamfer distance using gradient-descent
        # max_chamfer_distance = util.gradient_based_alignment(pre_splat_ply, post_splat_ply)

        # # Construct mesh and estimate volume difference
        # pre_splat_mesh_volume, post_splat_mesh_volume, volume_diff = util.mesh_construct_eval(pre_splat_ply, post_splat_ply)

        # # Calculate percentage volume loss
        # volume_loss_percentage = 0.0
        # if pre_splat_mesh_volume > 0:
        #     volume_loss_percentage = ((pre_splat_mesh_volume - post_splat_mesh_volume) / pre_splat_mesh_volume) * 100
        
        # Calculate percentage difference in bruised points
        # Access bruise values directly from the models
        pre_model = pre_pipeline.model
        post_model = post_pipeline.model
        
        # Get bruise values and apply sigmoid to convert to probabilities
        pre_bruise_probs = torch.sigmoid(pre_model.bruise).detach().cpu().numpy().squeeze()
        post_bruise_probs = torch.sigmoid(post_model.bruise).detach().cpu().numpy().squeeze()
        
        # Define threshold for bruised points (same as in the model)
        bruise_threshold = 0.25
        
        # Count bruised points (above threshold) - RAW MODEL DATA
        pre_bruised_count_raw = np.sum(pre_bruise_probs > bruise_threshold)
        post_bruised_count_raw = np.sum(post_bruise_probs > bruise_threshold)
        
        # Calculate total points - RAW MODEL DATA
        pre_total_points_raw = len(pre_bruise_probs)
        post_total_points_raw = len(post_bruise_probs)
        
        # Calculate percentages - RAW MODEL DATA
        pre_bruised_percentage_raw = (pre_bruised_count_raw / pre_total_points_raw) * 100 if pre_total_points_raw > 0 else 0.0
        post_bruised_percentage_raw = (post_bruised_count_raw / post_total_points_raw) * 100 if post_total_points_raw > 0 else 0.0
        
        # NOW CALCULATE FILTERED PERCENTAGES (matching what's actually exported)
        # Read the exported PLY files to get the actual filtered counts
        pre_splat_df = pd.read_csv(pre_splat_path.with_suffix('.csv')) if (pre_splat_path.with_suffix('.csv')).exists() else None
        post_splat_df = pd.read_csv(post_splat_path.with_suffix('.csv')) if (post_splat_path.with_suffix('.csv')).exists() else None
        
        # Parse PLY files manually to get bruise info from exported point clouds
        def get_bruise_info_from_ply(ply_path):
            """Extract bruise information from exported PLY file colors."""
            pcd = o3d.io.read_point_cloud(str(ply_path))
            points = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            
            if len(colors) == 0:
                return 0, len(points), 0.0
            
            # In the exporter, bruised points are colored purple (R=1, G=0, B=1)
            # Non-bruised are red (R=1, G=0, B=0)
            # Check for purple coloring (blue channel > 0.5 indicates bruised)
            bruised_mask = colors[:, 2] > 0.5  # Blue channel indicates bruising
            bruised_count = np.sum(bruised_mask)
            total_count = len(points)
            percentage = (bruised_count / total_count) * 100 if total_count > 0 else 0.0
            
            return bruised_count, total_count, percentage
        
        # Get filtered/exported bruise statistics
        pre_bruised_filtered, pre_total_filtered, pre_percentage_filtered = get_bruise_info_from_ply(pre_splat_path)
        post_bruised_filtered, post_total_filtered, post_percentage_filtered = get_bruise_info_from_ply(post_splat_path)
        
        # Calculate percentage difference in bruised points (both raw and filtered)
        bruised_points_percentage_diff_raw = post_bruised_percentage_raw - pre_bruised_percentage_raw
        bruised_points_percentage_diff_filtered = post_percentage_filtered - pre_percentage_filtered

        # CONSOLE.print(f"[bold green]Volume Loss Percentage: {volume_loss_percentage:.2f}%")
        CONSOLE.print(f"[bold cyan]RAW MODEL DATA (all gaussians):")
        CONSOLE.print(f"[bold cyan]  Pre-splat Bruised Points: {pre_bruised_count_raw}/{pre_total_points_raw} ({pre_bruised_percentage_raw:.2f}%)")
        CONSOLE.print(f"[bold cyan]  Post-splat Bruised Points: {post_bruised_count_raw}/{post_total_points_raw} ({post_bruised_percentage_raw:.2f}%)")
        CONSOLE.print(f"[bold cyan]  Bruised Points Percentage Difference: {bruised_points_percentage_diff_raw:.2f}%")
        CONSOLE.print(f"[bold yellow]FILTERED/EXPORTED DATA (after PLY export filtering):")
        CONSOLE.print(f"[bold yellow]  Pre-splat Bruised Points: {pre_bruised_filtered}/{pre_total_filtered} ({pre_percentage_filtered:.2f}%)")
        CONSOLE.print(f"[bold yellow]  Post-splat Bruised Points: {post_bruised_filtered}/{post_total_filtered} ({post_percentage_filtered:.2f}%)")
        CONSOLE.print(f"[bold yellow]  Bruised Points Percentage Difference: {bruised_points_percentage_diff_filtered:.2f}%")

        # Get the output and define the names to save to
        benchmark_info = {
            "Pre-Manipulation Splat": {
                "experiment_name": pre_config.experiment_name,
                "method_name": pre_config.method_name,
                "checkpoint": str(pre_checkpoint_path),
                # "results": pre_metrics_dict,
            },
            "Post-Manipulation Splat": {
                "experiment_name": post_config.experiment_name,
                "method_name": post_config.method_name,
                "checkpoint": str(post_checkpoint_path),
                # "results": post_metrics_dict,
            },
            # "Max_Chamfer_Distance": max_chamfer_distance,
            # "Pre-Splat Mesh Volume": pre_splat_mesh_volume,
            # "Post-Splat Mesh Volume": post_splat_mesh_volume,
            # "Mesh Volume Difference": volume_diff,
            # "Volume Loss Percentage": volume_loss_percentage,
            "Raw_Model_Data": {
                "Pre-Splat Bruised Points Count": int(pre_bruised_count_raw),
                "Post-Splat Bruised Points Count": int(post_bruised_count_raw),
                "Pre-Splat Total Points": int(pre_total_points_raw),
                "Post-Splat Total Points": int(post_total_points_raw),
                "Pre-Splat Bruised Points Percentage": pre_bruised_percentage_raw,
                "Post-Splat Bruised Points Percentage": post_bruised_percentage_raw,
                "Bruised Points Percentage Difference": bruised_points_percentage_diff_raw
            },
            "Filtered_Export_Data": {
                "Pre-Splat Bruised Points Count": int(pre_bruised_filtered),
                "Post-Splat Bruised Points Count": int(post_bruised_filtered),
                "Pre-Splat Total Points": int(pre_total_filtered),
                "Post-Splat Total Points": int(post_total_filtered),
                "Pre-Splat Bruised Points Percentage": pre_percentage_filtered,
                "Post-Splat Bruised Points Percentage": post_percentage_filtered,
                "Bruised Points Percentage Difference": bruised_points_percentage_diff_filtered
            },
            # Keep backwards compatibility with old field names (using raw data)
            "Pre-Splat Bruised Points Count": int(pre_bruised_count_raw),
            "Post-Splat Bruised Points Count": int(post_bruised_count_raw),
            "Pre-Splat Bruised Points Percentage": pre_bruised_percentage_raw,
            "Post-Splat Bruised Points Percentage": post_bruised_percentage_raw,
            "Bruised Points Percentage Difference": bruised_points_percentage_diff_raw
        }

        
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    print("\n")
    print("================================================")
    print("======== Running Bruisefacto Evaluation ========")
    print("================================================")
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeBruiseMetrics).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputeBruiseMetrics)  # noqa

