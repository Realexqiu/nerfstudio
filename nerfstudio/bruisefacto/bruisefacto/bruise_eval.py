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

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.scripts.exporter import ExportGaussianSplat
from nerfstudio.bruisefacto.bruisefacto import bruise_eval_utils as util

## To run bruise_eval.py, use the following command: ns-bruise-eval --config-pre /path/to/config_pre.yaml --config-post /path/to/config_post.yaml --output-path /path/to/output.json

@dataclass
class ComputeBruiseMetrics:
    """Load a checkpoint, compute some Bruise metrics, and save it to a JSON file."""

    # Path to config YAML file for pre and post splats. Tyro automatically creates CLI arguments from class attributes
    config_pre: Path
    config_post: Path

    # Name of the output .json file.
    output_path: Path = Path("bruise_metrics.json")

    # Output directory
    output_dir: Path = output_path.parent

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

        # Get the average evaluation image metrics
        pre_metrics_dict = pre_pipeline.get_average_eval_image_metrics(output_path=self.output_dir, get_std=True)
        post_metrics_dict = post_pipeline.get_average_eval_image_metrics(output_path=self.output_dir, get_std=True)

        import pdb; pdb.set_trace()

        # Export point clouds for pre and post splats
        pre_gs_exp = ExportGaussianSplat(load_config=self.config_pre, output_dir=self.output_dir, output_filename="pre_splat.ply")
        post_gs_exp = ExportGaussianSplat(load_config=self.config_post, output_dir=self.output_dir, output_filename="post_splat.ply")
        pre_gs_exp.main()
        post_gs_exp.main()

        # Define point cloud paths
        pre_splat_path = self.output_dir / "pre_splat.ply"
        post_splat_path = self.output_dir / "post_splat.ply"

        # Read and visualize the point clouds from the output directory
        pre_splat_ply_dataframe = util.load_ply_binary(pre_splat_path)
        post_splat_ply_dataframe = util.load_ply_binary(post_splat_path)
        pre_splat_ply = util.filter_point_cloud_from_dataframe(pre_splat_ply_dataframe)
        post_splat_ply = util.filter_point_cloud_from_dataframe(post_splat_ply_dataframe)
        util.show_point_cloud(pre_splat_ply)
        util.show_point_cloud(post_splat_ply)

        # Align point clouds and calculate max chamfer distance using gradient-descent
        max_chamfer_distance = util.gradient_based_alignment(pre_splat_ply, post_splat_ply)

        # Construct mesh and estimate volume difference
        pre_splat_mesh_volume, post_splat_mesh_volume, volume_diff = util.mesh_construct_eval(pre_splat_ply, post_splat_ply)

        # Get the output and define the names to save to
        benchmark_info = {
            "Pre-Manipulation Splat": {
                "experiment_name": pre_config.experiment_name,
                "method_name": pre_config.method_name,
                "checkpoint": str(pre_checkpoint_path),
                "results": pre_metrics_dict,
            },
            "Post-Manipulation Splat": {
                "experiment_name": post_config.experiment_name,
                "method_name": post_config.method_name,
                "checkpoint": str(post_checkpoint_path),
                "results": post_metrics_dict,
            },
            "Max_Chamfer_Distance": max_chamfer_distance,
            "Pre-Splat Mesh Volume": pre_splat_mesh_volume,
            "Post-Splat Mesh Volume": post_splat_mesh_volume,
            "Mesh Volume Difference": volume_diff
        }

        
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    print("Running bruise_eval.py script.")
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputeBruiseMetrics).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputeBruiseMetrics)  # noqa

