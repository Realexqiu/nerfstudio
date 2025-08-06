import numpy as np
import copy
import plotly.graph_objects as go
import pandas as pd
import struct
import trimesh
import open3d as o3d
import torch
from pathlib import Path
from typing import Union, Optional

def mesh_construct_eval(pcd1_np, pcd2_np):
    """
    Constructs watertight meshes from two point clouds and estimates their volumes.
    Computes the volume difference between the two meshes.

    Args:
        pcd1_np (np.ndarray): Numpy array representing the first point cloud.
        pcd2_np (np.ndarray): Numpy array representing the second point cloud.
    
    Returns:
        tuple: (float, float, float)
            - Volume of the first mesh.
            - Volume of the second mesh.
            - Absolute volume difference between the two meshes.
    """
    # Convert to open3d point cloud
    pcd1 = numpy_to_o3d_pcd(pcd1_np)
    pcd2 = numpy_to_o3d_pcd(pcd2_np)

    # Process both pcd1 and pcd2
    pcd1_processed, pcd1_hull_mesh, pcd1_hull_lineset, pcd1_hull_pcd = process_point_cloud(pcd1)
    pcd2_processed, pcd2_hull_mesh, pcd2_hull_lineset, pcd2_hull_pcd = process_point_cloud(pcd2)

    # Visualize outputs
    visualize_geometry(pcd1_processed, title="Clustered Point Cloud 1 (DBSCAN)")
    visualize_geometry(pcd1_hull_mesh, title="Convex Hull Mesh 1")
    visualize_geometry(pcd2_processed, title="Clustered Point Cloud 2 (DBSCAN)")
    visualize_geometry(pcd2_hull_mesh, title="Convex Hull Mesh 2")


    # pcd1_mesh = find_watertight_mesh(pcd1_hull_mesh, min_alpha=0.01, max_alpha=0.1, step=0.001)
    # pcd2_mesh = find_watertight_mesh(pcd2_hull_mesh, min_alpha=0.01, max_alpha=0.1, step=0.001)

    # pcd1_volume = estimate_mesh_volume(pcd1_mesh) if pcd1_mesh else 0
    # pcd2_volume = estimate_mesh_volume(pcd2_mesh) if pcd2_mesh else 0

    pcd1_volume = estimate_mesh_volume(pcd1_hull_mesh)
    pcd2_volume = estimate_mesh_volume(pcd2_hull_mesh)

    volume_difference = np.abs(pcd1_volume - pcd2_volume)
    
    return pcd1_volume, pcd2_volume, volume_difference

def process_point_cloud(pcd):
    """
    Processes a given point cloud by applying the following steps:
    1. Removes statistical outliers.
    2. Uses DBSCAN clustering to retain the largest cluster.
    3. Computes the convex hull of the filtered point cloud.
    4. Converts the convex hull mesh into a point cloud.
    
    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
    
    Returns:
        tuple: (clustered_pcd, hull_mesh hull_lineset, hull_pcd)
            clustered_pcd (o3d.geometry.PointCloud): The largest cluster after filtering.
            hull_mesh (o3d.geometry.Trianglemesh): Convex hull mesh.
            hull_lineset (o3d.geometry.LineSet): The convex hull as a line set.
            hull_pcd (o3d.geometry.PointCloud): The convex hull converted into a point cloud.
    """
    # Remove statistical outliers
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2)
    pcd_filtered = pcd.select_by_index(ind)
    
    # Use DBSCAN for finding the largest cluster to keep
    labels = np.array(pcd_filtered.cluster_dbscan(eps=0.02, min_points=40))
    max_label = labels.max()
    counts = [(labels == i).sum() for i in range(max_label + 1)]
    main_cluster = np.argmax(counts)
    indices = np.where(labels == main_cluster)[0]
    clustered_pcd = pcd_filtered.select_by_index(indices)
    
    # Compute Convex Hull
    hull_mesh, _ = clustered_pcd.compute_convex_hull()
    hull_lineset = o3d.geometry.LineSet.create_from_triangle_mesh(hull_mesh)
    
    # Convert the convex hull mesh to a point cloud
    hull_pcd = o3d.geometry.PointCloud()
    hull_pcd.points = hull_mesh.vertices
    
    return clustered_pcd, hull_mesh, hull_lineset, hull_pcd

def visualize_geometry(geometry, title="Open3D", window_size=(800, 600)):
    """
    Visualizes a given geometry in an Open3D window with a specified title.
    
    Args:
        geometry (o3d.geometry.Geometry): The geometry to display.
        title (str): The title for the visualization window.
        window_size (tuple): Size of the window.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=window_size[0], height=window_size[1])
    vis.add_geometry(geometry)
    vis.run()
    vis.destroy_window()

def estimate_mesh_volume(mesh):
    """
    Estimates the volume of a closed mesh using Open3D.0.687471

    Returns:
        float: The estimated volume of the mesh.
    """
    # Ensure the mesh is watertight (manifold)
    if not mesh.is_watertight():
        print("Warning: Mesh is not watertight! Volume estimation may be inaccurate.")
    
    # Compute the volume using Open3D's built-in method
    volume = mesh.get_volume()
    
    return volume

def find_watertight_mesh(pcd, min_alpha=0.01, max_alpha=0.1, step=0.001, save_path="watertight_mesh.ply"):
    """
    Iterates through alpha values until a watertight mesh is found.
    
    Args:
        pcd (open3d.geometry.PointCloud): Input point cloud.
        min_alpha (float): Starting alpha value.
        max_alpha (float): Maximum alpha value to test.
        step (float): Increment step for alpha values.
        save_path (str): File path to save the resulting watertight mesh.
    
    Returns:
        open3d.geometry.TriangleMesh: Watertight mesh if found, else None.
    """
    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    print("Searching for watertight mesh...")
    for alpha in np.arange(min_alpha, max_alpha, step):
        # print(f"Trying alpha = {alpha:.4f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)

        if mesh.is_watertight():
            print(f"✅ Found watertight mesh with alpha = {alpha:.4f}")
            o3d.io.write_triangle_mesh(save_path, mesh)
            print(f"Mesh saved to {save_path}\n")
            return mesh

    print("❌ No watertight mesh found in the given alpha range.")
    return None

def gradient_based_alignment(pcd1_np, pcd2_np):
    pcd1 = numpy_to_o3d_pcd(pcd1_np)
    pcd2 = numpy_to_o3d_pcd(pcd2_np)
    voxel_size = 0.025
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    pcd2_down = pcd2.voxel_down_sample(voxel_size)
    pcd1_down = normalize_pcd(pcd1_down)
    pcd2_down = normalize_pcd(pcd2_down)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pcd1_points = torch.from_numpy(np.asarray(pcd1_down.points)).float().to(device)
    pcd2_points = torch.from_numpy(np.asarray(pcd2_down.points)).float().to(device)

    # Variables to track the best transformation
    best_loss = float('inf')
    best_rot_params = None
    best_trans_params = None
    best_scale_param = None  # New addition

    num_runs = 50  # Number of times to run the optimization
    num_iterations = 200  # Number of iterations in each optimization

    for run in range(num_runs):
        print(f"\n=== Run {run+1}/{num_runs} ===")
        # Random initialization of transformation parameters
        rot_params = (torch.rand(3) * 2 * np.pi - np.pi).to(device)  # Random angles between -π and π
        rot_params.requires_grad = True
        trans_params = (torch.rand(3) * 2 - 1).to(device)  # Random translations between -1 and 1
        trans_params.requires_grad = True
        # Random initialization of scale parameter (log scale)
        scale_param = (torch.rand(1) * 0.4 - 0.2).to(device)  # Log scale between -0.2 and 0.2
        scale_param.requires_grad = True

        # Define the optimizer with per-parameter learning rates
        optimizer = torch.optim.Adam([{'params': [rot_params, trans_params], 'lr': 0.05}, {'params': [scale_param], 'lr': 0.005}])

        for i in range(num_iterations):
            optimizer.zero_grad()
            angle = torch.norm(rot_params)
            if angle.item() == 0:
                rot_matrix = torch.eye(3, device=device)
            else:
                axis = rot_params / angle
                K = torch.tensor([[0, -axis[2], axis[1]],
                                [axis[2], 0, -axis[0]],
                                [-axis[1], axis[0], 0]], device=device)
                rot_matrix = torch.eye(3, device=device) + torch.sin(angle) * K + \
                            (1 - torch.cos(angle)) * torch.matmul(K, K)
            # Compute the scale
            scale = torch.exp(scale_param)
            # Apply transformation
            transformed_pcd1 = scale * torch.matmul(pcd1_points, rot_matrix.t()) + trans_params
            loss = chamfer_distance(transformed_pcd1, pcd2_points)
            loss.backward()
            optimizer.step()
            if (i + 1) % 2 == 0:
                print(f"Iteration {i+1}/{num_iterations}, Loss: {loss.item():.6f}", end='\r')

        final_loss = loss.item() # type: ignore
        print(f"Final Loss for run {run+1}: {final_loss:.6f}")

        # Check if this run has the best loss so far
        if final_loss < best_loss:
            best_loss = final_loss
            best_rot_params = rot_params.detach().clone()
            best_trans_params = trans_params.detach().clone()
            best_scale_param = scale_param.detach().clone()
            print(f"New best loss found: {best_loss:.6f}")

    # Apply the best transformation found
    with torch.no_grad():
        # Compute the best scale
        scale = torch.exp(best_scale_param)
        
        # Compute rotation matrix
        angle = torch.norm(best_rot_params)
        if angle.item() == 0:
            rot_matrix = torch.eye(3, device=device)
        else:
            axis = best_rot_params / angle
            K = torch.tensor([[0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0]], device=device)
            rot_matrix = torch.eye(3, device=device) + torch.sin(angle) * K + \
                        (1 - torch.cos(angle)) * torch.matmul(K, K)
        # Apply the best transformation
        transformed_pcd1 = scale * torch.matmul(pcd1_points, rot_matrix.t()) + best_trans_params

    transformed_points_np = transformed_pcd1.cpu().numpy()
    pcd2_np = pcd2_points.cpu().numpy()

    # Visualize the aligned point clouds
    show_two_point_clouds(transformed_points_np, pcd2_np)

    return best_loss

def chamfer_distance(pcd1, pcd2):
    x, y = pcd1.unsqueeze(1), pcd2.unsqueeze(0)
    diff = x - y
    dist = torch.sum(diff ** 2, dim=2)
    dist1 = torch.min(dist, dim=1)[0]
    dist2 = torch.min(dist, dim=0)[0]
    return torch.mean(dist1) + torch.mean(dist2)

def show_two_point_clouds(pc1, pc2, title="Aligned Point Clouds"):
    scatter1 = go.Scatter3d(
        x=pc1[:, 0], y=pc1[:, 1], z=pc1[:, 2],
        mode='markers',
        marker=dict(size=2, color='red'),
        name='Transformed Source'
    )

    scatter2 = go.Scatter3d(
        x=pc2[:, 0], y=pc2[:, 1], z=pc2[:, 2],
        mode='markers',
        marker=dict(size=2, color='blue'),
        name='Target'
    )

    layout = go.Layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
            aspectmode='data'
        ),
        width=800,
        height=600
    )

    fig = go.Figure(data=[scatter1, scatter2], layout=layout)
    fig.show()

def normalize_pcd(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    points /= max_dist
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def numpy_to_o3d_pcd(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def load_ply_binary(ply_path):
    """Reads a binary PLY file and extracts all attributes as a DataFrame."""
    with open(ply_path, "rb") as f:
        # Read header
        header = []
        while True:
            line = f.readline().strip().decode("utf-8")
            header.append(line)
            if line == "end_header":
                break
        
        # Extract property names from header
        properties = []
        for line in header:
            if line.startswith("property"):
                properties.append(line.split()[-1])
        
        # Read binary data
        dtype_format = "<" + "f" * len(properties)  # Little-endian floats
        struct_size = struct.calcsize(dtype_format)
        data = []
        
        while True:
            binary_chunk = f.read(struct_size)
            if not binary_chunk:
                break
            data.append(struct.unpack(dtype_format, binary_chunk))
    
    # Convert to DataFrame
    df = pd.DataFrame(data, columns=properties)
    return df

def load_ply_with_open3d(path: Path) -> pd.DataFrame:
    # read_point_cloud auto‑detects ASCII vs binary
    pcd = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pcd.points)                # (N,3) float64 array
    df = pd.DataFrame(pts, columns=["x","y","z"])
    # if you need colors or normals, you can do:
    # if pcd.has_colors(): df[["r","g","b"]] = np.asarray(pcd.colors)
    # if pcd.has_normals(): df[["nx","ny","nz"]] = np.asarray(pcd.normals)
    return df

def filter_point_cloud_from_dataframe(df: pd.DataFrame, min: Optional[np.ndarray] = None, max: Optional[np.ndarray] = None):
    """
    Filters point cloud data from a dataframe based on predefined XYZ thresholds.

    Args:
        df (pd.DataFrame): A dataframe containing columns ['x', 'y', 'z'] representing point cloud coordinates.
        min (np.ndarray, optional): A numpy array of shape (3,) representing the minimum XYZ thresholds. Defaults to None.
        max (np.ndarray, optional): A numpy array of shape (3,) representing the maximum XYZ thresholds. Defaults to None.

    Returns:
        np.ndarray: A numpy array of shape (n, 3) containing the filtered or unfiltered point cloud.
    """
    # Extract XYZ points from the dataframe
    original_points = df[['x', 'y', 'z']].to_numpy()

    # If no filtering bounds are provided, return the original point cloud
    if min is None or max is None:
        print(f"No filtering applied. Returning original point cloud with {original_points.shape[0]} points")
        return original_points

    # Define filtering conditions based on thresholds
    condition_x = (original_points[:, 0] > min[0]) & (original_points[:, 0] < max[0])
    condition_y = (original_points[:, 1] > min[1]) & (original_points[:, 1] < max[1])
    condition_z = (original_points[:, 2] > min[2]) & (original_points[:, 2] < max[2])

    # Apply the filtering conditions
    filtered_points = original_points[condition_x & condition_y & condition_z]

    # Display the filtering results
    print(f"Original point cloud has {original_points.shape[0]} points")
    print(f"Filtered point cloud has {filtered_points.shape[0]} points")

    return filtered_points

def show_point_cloud(pcd, title: Optional[str] = None):
    """
    Displays a 3D point cloud using Plotly graph objects.

    Args:
        pcd (np.ndarray): A numpy array of shape (n, 3) containing the 3D points.
        title (str): The title of the plot.

    Returns:
        None: The function displays the point cloud visualization.
    """
    scatter = go.Scatter3d(x=pcd[:, 0], y=pcd[:, 1], z=pcd[:, 2], mode='markers', marker=dict(size=2, color='blue'))

    # Create layout
    layout = go.Layout(scene=dict(xaxis=dict(showgrid=False), yaxis=dict(showgrid=False), zaxis=dict(showgrid=False)))

    # Create the figure and show the plot
    fig1 = go.Figure(data=[scatter], layout=layout)
    fig1.update_layout(autosize=False, width=1000, height=400, title={'text': title}, margin=dict(l=0, r=0, b=0, t=50, pad=4), paper_bgcolor="Green")
    fig1.show()

def downsample_pcd(pcd, voxel_size):
    """
    Preprocesses a point cloud by downsampling and computing FPFH features.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        voxel_size (float): The size of the voxel for downsampling.

    Returns:
        tuple: (o3d.geometry.PointCloud, o3d.pipelines.registration.Feature)
            - The downsampled point cloud.
            - The computed FPFH feature of the point cloud.
    """
    # Downsample the point cloud
    pcd_downsampled = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    pcd_downsampled.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    radius_feature = voxel_size * 5 
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_downsampled, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    return pcd_downsampled, pcd_fpfh

def draw_registration_result(source, target, transformation):
    """
    Visualizes the registration result of two point clouds.

    Args:
        source (o3d.geometry.PointCloud): The source point cloud.
        target (o3d.geometry.PointCloud): The target point cloud.
        transformation (np.ndarray): The 4x4 transformation matrix applied to the source point cloud.

    Returns:
        None: Displays the registered point clouds using Open3D visualization.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp, coordinate_frame])



