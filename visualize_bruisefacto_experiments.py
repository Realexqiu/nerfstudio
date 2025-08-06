#!/usr/bin/env python3

"""
Script to visualize bruisefacto point clouds from all experiments.
Automatically finds the most recent run for each experiment in dt_splats folder.
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import open3d as o3d
from collections import OrderedDict

# Add the nerfstudio directory to Python path to import modules
current_dir = Path(__file__).parent
nerfstudio_path = current_dir / "nerfstudio"
if nerfstudio_path.exists():
    sys.path.insert(0, str(nerfstudio_path))

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.bruisefacto.bruisefacto.bruisefacto import BruisefactoModel
from nerfstudio.bruisefacto.bruisefacto.bruisefacto_exporter import Export3DGS


class BruisefactoVisualizer:
    def __init__(self, dt_splats_path: str = "/home/alex/dt_splats"):
        self.dt_splats_path = Path(dt_splats_path)
        self.experiments_data = {}
        
    def find_latest_runs(self) -> Dict[str, Path]:
        """Find the latest bruisefacto run for each experiment."""
        latest_runs = {}
        
        if not self.dt_splats_path.exists():
            CONSOLE.print(f"[bold red]Error: dt_splats path {self.dt_splats_path} does not exist!")
            return latest_runs
            
        for experiment_dir in self.dt_splats_path.iterdir():
            if not experiment_dir.is_dir():
                continue
                
            bruisefacto_dir = experiment_dir / "bruisefacto"
            if not bruisefacto_dir.exists():
                continue
                
            # Find all timestamp directories
            timestamp_dirs = []
            for timestamp_dir in bruisefacto_dir.iterdir():
                if timestamp_dir.is_dir() and self._is_valid_timestamp(timestamp_dir.name):
                    config_path = timestamp_dir / "config.yml"
                    model_dir = timestamp_dir / "nerfstudio_models"
                    if config_path.exists() and model_dir.exists():
                        timestamp_dirs.append(timestamp_dir)
            
            if timestamp_dirs:
                # Sort by timestamp to get the latest
                latest_dir = max(timestamp_dirs, key=lambda x: x.name)
                latest_runs[experiment_dir.name] = latest_dir
                CONSOLE.print(f"[green]Found latest run for {experiment_dir.name}: {latest_dir.name}")
                
        return latest_runs
    
    def _is_valid_timestamp(self, dirname: str) -> bool:
        """Check if directory name is a valid timestamp format."""
        try:
            datetime.strptime(dirname, "%Y-%m-%d_%H%M%S")
            return True
        except ValueError:
            return False
    
    def extract_point_cloud(self, config_path: Path, experiment_name: str) -> Optional[Tuple[np.ndarray, np.ndarray, Dict]]:
        """Extract point cloud from a bruisefacto model."""
        try:
            CONSOLE.print(f"[blue]Loading model for {experiment_name}...")
            
            # Load the model using eval_setup
            _, pipeline, _, _ = eval_setup(config_path, test_mode="inference")
            model: BruisefactoModel = pipeline.model
            
            with torch.no_grad():
                # Extract positions
                positions = model.means.cpu().numpy()
                count = positions.shape[0]
                
                # Extract colors and convert to 0-1 range
                colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
                
                # Initialize metadata dictionary
                metadata = {
                    'opacity': model.opacities.data.cpu().numpy(),
                    'scales': model.scales.data.cpu().numpy(),
                    'quats': model.quats.data.cpu().numpy(),
                    'experiment_name': experiment_name,
                    'total_gaussians': count
                }
                
                # Check for bruise information
                if hasattr(model, 'bruise'):
                    bruise_vals = torch.sigmoid(model.bruise).detach().cpu().numpy().squeeze(-1)
                    bruised_mask = bruise_vals > 0.25
                    
                    # Create special coloring for bruised regions
                    colors_viz = colors.copy()
                    # Non-bruised areas: red
                    colors_viz[:] = [1.0, 0.0, 0.0]  # Red
                    # Bruised areas: purple
                    colors_viz[bruised_mask] = [1.0, 0.0, 1.0]  # Purple/Magenta
                    
                    metadata['bruise_values'] = bruise_vals
                    metadata['bruised_mask'] = bruised_mask
                    metadata['bruised_count'] = np.sum(bruised_mask)
                    metadata['bruised_percentage'] = (np.sum(bruised_mask) / count) * 100
                    
                    CONSOLE.print(f"[green]Found bruise data: {np.sum(bruised_mask)}/{count} gaussians bruised ({metadata['bruised_percentage']:.2f}%)")
                    
                    return positions, colors_viz, metadata
                else:
                    CONSOLE.print(f"[yellow]No bruise data found for {experiment_name}")
                    return positions, colors, metadata
                    
        except Exception as e:
            CONSOLE.print(f"[bold red]Error loading model for {experiment_name}: {e}")
            return None
    
    def create_coordinate_frame(self, size: float = 0.5) -> o3d.geometry.TriangleMesh:
        """Create a coordinate frame for visualization."""
        return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)
    
    def visualize_point_cloud(self, positions: np.ndarray, colors: np.ndarray, 
                            metadata: Dict, show_coordinate_frame: bool = True):
        """Visualize a single point cloud with optional coordinate frame."""
        
        # Apply initial filtering to remove outliers
        filter_mask = (
            (positions[:, 0] >= -1.0) & (positions[:, 0] <= 1.0) &
            (positions[:, 1] >= -1.0) & (positions[:, 1] <= 1.0) &
            (positions[:, 2] >= -1.0) & (positions[:, 2] <= 1.0)
        )
        
        filtered_positions = positions[filter_mask]
        filtered_colors = colors[filter_mask]
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_positions)
        pcd.colors = o3d.utility.Vector3dVector(filtered_colors)
        
        # Prepare geometries for visualization
        geometries = [pcd]
        
        if show_coordinate_frame:
            coordinate_frame = self.create_coordinate_frame(size=0.5)
            geometries.append(coordinate_frame)
        
        # Print statistics
        experiment_name = metadata.get('experiment_name', 'Unknown')
        total_points = filtered_positions.shape[0]
        original_points = metadata.get('total_gaussians', 'Unknown')
        
        CONSOLE.print(f"[bold blue]Visualizing {experiment_name}")
        CONSOLE.print(f"[blue]Points after filtering: {total_points}/{original_points}")
        
        if 'bruised_percentage' in metadata:
            CONSOLE.print(f"[blue]Bruised regions: {metadata['bruised_percentage']:.2f}% (Purple = Bruised, Red = Healthy)")
        
        # Visualize
        CONSOLE.print(f"[green]Opening visualization window for {experiment_name}...")
        CONSOLE.print(f"[green]Close the window to continue to the next experiment.")
        
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Bruisefacto Point Cloud - {experiment_name}",
            width=1200,
            height=800
        )
    
    def visualize_all_experiments(self, interactive: bool = True, 
                                coordinate_frame: bool = True):
        """Visualize point clouds for all experiments."""
        latest_runs = self.find_latest_runs()
        
        if not latest_runs:
            CONSOLE.print("[bold red]No valid bruisefacto experiments found!")
            return
        
        CONSOLE.print(f"[bold green]Found {len(latest_runs)} experiments to visualize")
        
        for i, (experiment_name, run_path) in enumerate(latest_runs.items(), 1):
            CONSOLE.print(f"\n[bold yellow]Processing experiment {i}/{len(latest_runs)}: {experiment_name}")
            
            config_path = run_path / "config.yml"
            
            # Extract point cloud
            result = self.extract_point_cloud(config_path, experiment_name)
            if result is None:
                CONSOLE.print(f"[red]Skipping {experiment_name} due to errors")
                continue
                
            positions, colors, metadata = result
            
            # Visualize
            if interactive:
                self.visualize_point_cloud(positions, colors, metadata, coordinate_frame)
            else:
                # For non-interactive mode, just print stats
                total_points = positions.shape[0]
                CONSOLE.print(f"[blue]{experiment_name}: {total_points} gaussians")
                if 'bruised_percentage' in metadata:
                    CONSOLE.print(f"[blue]  Bruised: {metadata['bruised_percentage']:.2f}%")
    
    def export_all_point_clouds(self, output_dir: str = "./exported_point_clouds"):
        """Export all point clouds to PLY files for external visualization."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        latest_runs = self.find_latest_runs()
        
        for experiment_name, run_path in latest_runs.items():
            CONSOLE.print(f"[blue]Exporting {experiment_name}...")
            
            config_path = run_path / "config.yml"
            output_filename = f"{experiment_name}_point_cloud.ply"
            
            try:
                exporter = Export3DGS(
                    load_config=config_path,
                    output_dir=output_path,
                    output_filename=output_filename
                )
                
                # We need to modify the main method to skip interactive parts
                # This is a simplified export without interactive filtering
                self._export_simplified(exporter, experiment_name)
                
            except Exception as e:
                CONSOLE.print(f"[red]Error exporting {experiment_name}: {e}")
    
    def _export_simplified(self, exporter: "Export3DGS", experiment_name: str):
        """Simplified export without interactive components."""
        # This is a simplified version of the export logic
        _, pipeline, _, _ = eval_setup(exporter.load_config, test_mode="inference")
        model: BruisefactoModel = pipeline.model
        filename = exporter.output_dir / exporter.output_filename
        
        map_to_tensors = OrderedDict()
        
        with torch.no_grad():
            positions = model.means.cpu().numpy()
            count = positions.shape[0]
            
            map_to_tensors["x"] = positions[:, 0].astype(np.float32)
            map_to_tensors["y"] = positions[:, 1].astype(np.float32)
            map_to_tensors["z"] = positions[:, 2].astype(np.float32)
            
            colors = torch.clamp(model.colors.clone(), 0.0, 1.0).data.cpu().numpy()
            colors = (colors * 255).astype(np.uint8)
            map_to_tensors["red"] = colors[:, 0].astype(np.uint8)
            map_to_tensors["green"] = colors[:, 1].astype(np.uint8)
            map_to_tensors["blue"] = colors[:, 2].astype(np.uint8)
            
            if hasattr(model, 'bruise'):
                bruise_vals = torch.sigmoid(model.bruise).detach().cpu().numpy().squeeze(-1)
                bruised_mask = bruise_vals > 0.25
                
                # Apply coloring: red for healthy, purple for bruised
                map_to_tensors["red"] = np.full_like(map_to_tensors["red"], 255, dtype=np.uint8)
                map_to_tensors["green"] = np.full_like(map_to_tensors["green"], 0, dtype=np.uint8)
                map_to_tensors["blue"] = np.full_like(map_to_tensors["blue"], 0, dtype=np.uint8)
                map_to_tensors["blue"][bruised_mask] = 255
                
                CONSOLE.print(f"[green]{experiment_name}: {np.sum(bruised_mask)}/{count} bruised gaussians")
        
        # Apply basic filtering
        select = (
            (positions[:, 0] >= -1.0) & (positions[:, 0] <= 1.0) &
            (positions[:, 1] >= -1.0) & (positions[:, 1] <= 1.0) &
            (positions[:, 2] >= -1.0) & (positions[:, 2] <= 1.0)
        )
        
        for k in map_to_tensors.keys():
            map_to_tensors[k] = map_to_tensors[k][select]
        
        final_count = np.sum(select)
        
        # Save the point cloud
        exporter.write_ply(str(filename), final_count, map_to_tensors)
        CONSOLE.print(f"[green]Exported {experiment_name} to {filename}")


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize Bruisefacto experiment point clouds")
    parser.add_argument("--dt-splats-path", type=str, default="/home/alex/dt_splats",
                       help="Path to dt_splats directory")
    parser.add_argument("--mode", choices=["visualize", "export", "both"], default="visualize",
                       help="Mode: visualize interactively, export PLY files, or both")
    parser.add_argument("--output-dir", type=str, default="./exported_point_clouds",
                       help="Output directory for exported PLY files")
    parser.add_argument("--no-coordinate-frame", action="store_true",
                       help="Disable coordinate frame display")
    parser.add_argument("--non-interactive", action="store_true",
                       help="Run in non-interactive mode (just print statistics)")
    
    args = parser.parse_args()
    
    # Create visualizer
    visualizer = BruisefactoVisualizer(args.dt_splats_path)
    
    if args.mode in ["visualize", "both"]:
        CONSOLE.print("[bold green]Starting visualization of bruisefacto experiments...")
        visualizer.visualize_all_experiments(
            interactive=not args.non_interactive,
            coordinate_frame=not args.no_coordinate_frame
        )
    
    if args.mode in ["export", "both"]:
        CONSOLE.print(f"[bold green]Exporting point clouds to {args.output_dir}...")
        visualizer.export_all_point_clouds(args.output_dir)
    
    CONSOLE.print("[bold green]Done!")


if __name__ == "__main__":
    main() 