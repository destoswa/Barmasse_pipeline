import os
import numpy as np
import laspy
# import json
import scipy
import copy
from tqdm import tqdm
from time import time
import itertools

# ====== MODE ======
MODE = "PRE"    # choose between "PRE" (for preprocessing) and "POST" (for postprocessing)
# ==================

# ===== INPUTS =====
SRC_INPUT = r"D:\GitHubProjects\Terranum_repo\TreeSegmentation\data\Barmasse\Barmasse_2025\test_from_python_script\Barmasse_2025_AllScans_Raw_Sub2cm_clean.laz"
TILE_SIZE = 100
OVERLAP = 20
GRID_SIZE = 10
STRIPE_WIDTH = 20
SHIFT=0
METHOD='quadric'
do_save_floor=True
do_keep_existing_flatten = True
# ==================


# def tilling(src_input, src_target, tile_size, overlap=0, verbose=True):
#     """
#     Tile the input LiDAR file into square tiles using PDAL and store them in the destination folder.

#     Args:
#         - verbose (bool): Whether to print verbose status updates.

#     Returns:
#         - None: Splits the input file into tiles and saves them in the destination folder.
#     """

#     print(f"Start tilling (with overlap = {overlap}m)...")
#     os.makedirs(src_target, exist_ok=True)
    
#     # compute the estimate number of tiles
#     if verbose:
#         print("Computing the bounds...")
#     original_file = laspy.read(src_input)
#     x_min = original_file.x.min()
#     x_max = original_file.x.max()
#     y_min = original_file.y.min()
#     y_max = original_file.y.max() 
#     if verbose:
#         print('Done!')

#     output_pattern = os.path.join(
#         src_target, 
#         os.path.basename(src_input).split('.')[0] + "_tile_#.laz",
#         )

#     x_steps = int((x_max - x_min) / tile_size) + 1
#     y_steps = int((y_max - y_min) / tile_size) + 1
#     combinations = list(itertools.product(range(x_steps), range(y_steps)))
#     list_bounds = []
#     for (i,j) in combinations:
#         x0 = x_min + i * tile_size - overlap
#         x1 = x_min + (i + 1) * tile_size + overlap
#         y0 = y_min + j * tile_size - overlap
#         y1 = y_min + (j + 1) * tile_size + overlap

#         bounds = f"([{x0},{x1}],[{y0},{y1}])"
#         list_bounds.append(bounds)

#     pipeline_json = {
#         "pipeline": [
#             {
#                 "type": "readers.las",
#                 "filename": src_input,
#                 "spatialreference": "EPSG:2056",
#                 "extra_dims": "id_point=uint32"
#             },
#             {
#                 "type": "filters.crop", 
#                 "bounds": list_bounds
#             },
#             {
#                 "type": "writers.las", 
#                 "filename": output_pattern, 
#                 'extra_dims': "id_point=uint32"
#             }
#         ]
#     }
    
#     if verbose:
#         print("Creation of the tiles (might take a few minutes)")
#     pipeline = pdal.Pipeline(json.dumps(pipeline_json))
#     pipeline.execute()
#     if verbose:
#         print("Process done")

def tilling(src_input, src_target, tile_size, overlap=0, shift=0, verbose=True):
    """
    Crops a LAS/LAZ file into tiles using laspy directly (preserves id_point and coordinates).
    """
    os.makedirs(src_target, exist_ok=True)

    las = laspy.read(src_input)
    x_min, x_max = las.x.min() - shift, las.x.max()
    y_min, y_max = las.y.min() - shift, las.y.max()

    x_steps = int((x_max - x_min) / tile_size) + 1
    y_steps = int((y_max - y_min) / tile_size) + 1

    combinations = list(itertools.product(range(x_steps), range(y_steps)))
    
    print("Creation of tiles:")
    for _, (ix, iy) in tqdm(enumerate(combinations), total=len(combinations)):
    # for ix in range(x_steps):
    #     for iy in range(y_steps):
        x0 = x_min + ix * tile_size - overlap
        x1 = x_min + (ix + 1) * tile_size + overlap
        y0 = y_min + iy * tile_size - overlap
        y1 = y_min + (iy + 1) * tile_size + overlap

        mask = (
            (las.x >= x0) & (las.x <= x1) &
            (las.y >= y0) & (las.y <= y1)
        )
        if not np.any(mask):
            continue

        # ✅ Create new file with proper header scale/offset
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        header.offsets = las.header.offsets
        header.scales = las.header.scales

        # ✅ Copy CRS if any
        if hasattr(las.header, "epsg") and las.header.epsg is not None:
            header.epsg = las.header.epsg

        tile = laspy.LasData(header)
        tile.points = las.points[mask]

        tile_filename = os.path.join(
            src_target,
            f"{os.path.splitext(os.path.basename(src_input))[0]}_tile_{ix}_{iy}.laz"
        )
        tile.write(tile_filename)


def stripes_file(src_input_file, src_output, dims, do_keep_existing=False, verbose=True):
    """
    Crops a LAS/LAZ file into tiles using laspy directly (preserves id_point and coordinates).
    """
    [tile_size_x, tile_size_y] = dims
    las = laspy.read(src_input_file)
    os.makedirs(src_output, exist_ok=True)

    xmin, xmax, ymin, ymax = np.min(las.x), np.max(las.x), np.min(las.y), np.max(las.y)

    x_edges = np.arange(xmin, xmax, tile_size_x)
    y_edges = np.arange(ymin, ymax, tile_size_y)

    combinations = list(itertools.product(x_edges, y_edges))
    # for (x0, y0) in combinations:
    num_skipped = 0
    for id_stripe, (x0, y0) in tqdm(enumerate(combinations), total=len(combinations)):
        x1 = x0 + tile_size_x
        y1 = y0 + tile_size_y

        mask = (
            (las.x >= x0) & (las.x <= x1) &
            (las.y >= y0) & (las.y <= y1)
        )
        if not np.any(mask):
            num_skipped += 1
            continue

        # ✅ Create new file with proper header scale/offset
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        header.offsets = las.header.offsets
        header.scales = las.header.scales

        # ✅ Copy CRS if any
        if hasattr(las.header, "epsg") and las.header.epsg is not None:
            header.epsg = las.header.epsg

        tile = laspy.LasData(header)
        tile.points = las.points[mask]

        tile_filename = os.path.join(
            src_output,
            f"{os.path.splitext(os.path.basename(src_input_file))[0]}_stripe_{id_stripe - num_skipped}.laz"
        )
        tile.write(tile_filename)


def remove_duplicates(laz_file, decimals=2):
    """
    Removes duplicate points from a LAS/LAZ file based on rounded 3D coordinates.

    Args:
        - laz_file (laspy.LasData): Input LAS/LAZ file as a laspy object.
        - decimals (int, optional): Number of decimals to round the coordinates for duplicate detection. Defaults to 2.

    Returns:
        - laspy.LasData: A new laspy object with duplicate points removed.
    """
        
    coords = np.round(np.vstack((laz_file.x, laz_file.y, laz_file.z)).T, decimals)
    _, unique_indices = np.unique(coords, axis=0, return_index=True)
    mask = np.zeros(len(coords), dtype=bool)
    mask[unique_indices] = True

    # Create new LAS object
    header = laspy.LasHeader(point_format=laz_file.header.point_format, version=laz_file.header.version)
    new_las = laspy.LasData(header)

    for dim in laz_file.point_format.dimension_names:
        setattr(new_las, dim, getattr(laz_file, dim)[mask])

    return new_las


def match_pointclouds(laz1, laz2):
    """Sort laz2 to match the order of laz1 without changing laz1's order.

    Args:
        laz1: laspy.LasData object (reference order)
        laz2: laspy.LasData object (to be sorted)
    
    Returns:
        laz2 sorted to match laz1
    """
    # Retrieve and round coordinates for robust matching
    coords_1 = np.round(np.vstack((laz1.x, laz1.y, laz1.z)), 2).T
    coords_2 = np.round(np.vstack((laz2.x, laz2.y, laz2.z)), 2).T

    # Verify laz2 is of the same size as laz1
    assert len(coords_2) == len(coords_1), "laz2 should be a subset of laz1"

    # Create a dictionary mapping from coordinates to indices
    coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coords_1)}

    # Find indices in laz1 that correspond to laz2
    matching_indices = []
    failed = 0
    for coord in coords_2:
        try:
            matching_indices.append(coord_to_idx[tuple(coord)])
        except Exception as e:
            failed += 1

    matching_indices = np.array([coord_to_idx[tuple(coord)] for coord in coords_2])

    # Sort laz2 to match laz1
    sorted_indices = np.argsort(matching_indices)

    # Apply sorting to all attributes of laz2
    laz2.points = laz2.points[sorted_indices]

    return laz2  # Now sorted to match laz1


# def flattening_tile(tile_src, tile_new_original_src, grid_size=10, do_save_floor=False, do_keep_existing=False, do_extrapolate_outside_hull=False, verbose=True):
#     """
#     Flattens a tile by interpolating the ground surface and subtracting it from the original elevation.

#     Args:
#         - tile_src (str): Path to the input tile in LAS/LAZ format.
#         - tile_new_original_src (str): Path to save the resized original tile after filtering.
#         - grid_size (int, optional): Size of the grid in meters for local interpolation. Defaults to 10.
#         - verbose (bool, optional): Whether to display progress and debug information. Defaults to True.

#     Returns:
#         - None: Saves the floor and flattened versions of the tile and updates the original file.
#     """
#     if os.path.exists(tile_new_original_src) and do_keep_existing:
#         if verbose:
#             print(f"Skipping. {tile_new_original_src} exists already")
#         return
    
#     # Load file
#     laz = laspy.read(tile_src)
#     init_len = len(laz)
#     if init_len == 0:
#         return
    
#     if verbose:
#         print(f"Removing duplicates: From {init_len} to {len(laz)}")
    
#     points = np.vstack((laz.x, laz.y, laz.z)).T
#     points_flatten = copy.deepcopy(points)
#     points_interpolated = copy.deepcopy(points)

#     # Divide into tiles and find local minimums
#     x_min, y_min = np.min(points[:, :2], axis=0)
#     x_max, y_max = np.max(points[:, :2], axis=0)

#     x_bins = np.append(np.arange(x_min, x_max, grid_size), x_max)
#     y_bins = np.append(np.arange(y_min, y_max, grid_size), y_max)

#     grid = {i:{j:[] for j in range(y_bins.size - 1)} for i in range(x_bins.size -1)}
#     for _, (px, py, pz) in tqdm(enumerate(points), total=len(points), desc="Creating grid", disable=verbose==False):
#         xbin = np.clip(0, (px - x_min) // grid_size, x_bins.size - 2)
#         ybin = np.clip(0, (py - y_min) // grid_size, y_bins.size - 2)
#         try:
#             grid[xbin][ybin].append((px, py, pz))
#         except Exception as e:
#             print(xbin)
#             print(ybin)
#             print(x_bins)
#             print(y_bins)
#             print(grid.keys())
#             print(grid[0].keys())
#             raise e


#     # Create grid_min
#     grid_used = np.zeros((x_bins.size - 1, y_bins.size - 1))
#     lst_grid_min = []
#     lst_grid_min_pos = []
#     for x in grid.keys():
#         for y in grid[x].keys():
#             if np.array(grid[x][y]).shape[0] > 0:
#                 grid_used[x, y] = 1
#                 lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
#                 arg_min = np.argmin(np.array(grid[x][y])[:,2])
#                 lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2])

#                 # test if border
#                 if x == list(grid.keys())[0]:
#                     lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
#                     lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2] - [5, 0])
#                 if x == list(grid.keys())[-1]:
#                     lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
#                     lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2] + [5, 0])
#                 if y == list(grid[x].keys())[0]:
#                     lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
#                     lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2] - [0, 5])
#                 if y == list(grid[x].keys())[-1]:
#                     lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
#                     lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2] + [0, 5])
#             else:
#                 grid_used[x, y] = 0
#     arr_grid_min_pos = np.vstack(lst_grid_min_pos)
#     if verbose:
#         print("Resulting grid:")
#         print(arr_grid_min_pos.shape)
#         print(grid_used)

#     # Interpolate
#     points_xy = np.array(points)[:,0:2]
#     interpolated_min_z = scipy.interpolate.griddata(arr_grid_min_pos, np.array(lst_grid_min), points_xy, method="cubic", fill_value=-1)

#     # Fill NaNs with nearest neighbor interpolation
#     if do_extrapolate_outside_hull:
#         nan_mask = interpolated_min_z == -1
#         x = np.array(points)[:,0]
#         y = np.array(points)[:,1]

#         if np.any(nan_mask):
#             interpolated_min_z[nan_mask] = scipy.interpolate.griddata(arr_grid_min_pos, np.array(lst_grid_min), (x[nan_mask], y[nan_mask]), method='nearest')

#     mask_valid = np.array([x != -1 for x in list(interpolated_min_z)])
#     points_interpolated = points_interpolated[mask_valid]
#     points_interpolated[:, 2] = interpolated_min_z[mask_valid]

#     if verbose:
#         print("Interpolation:")
#         print(f"Original number of points: {points.shape[0]}")
#         print(f"Interpollated number of points: {points_interpolated.shape[0]} ({int(points_interpolated.shape[0] / points.shape[0]*100)}%)")

#     # save floor
#     filtered_points = {dim: getattr(laz, dim)[mask_valid] for dim in laz.point_format.dimension_names}
#     header = laspy.LasHeader(point_format=laz.header.point_format, version=laz.header.version)
#     new_las = laspy.LasData(header)

#     #   _Assign filtered and modified data
#     for dim, values in filtered_points.items():
#         setattr(new_las, dim, values)
#     setattr(new_las, 'x', points_interpolated[:,0])
#     setattr(new_las, 'y', points_interpolated[:,1])
#     setattr(new_las, 'z', points_interpolated[:,2])

#     #   _Save new file
#     if do_save_floor:
#         new_las.write(tile_new_original_src.split('.laz')[0] + "_floor.laz")
#         if verbose:
#             print("Saved file: ", tile_new_original_src.split('.laz')[0] + "_floor.laz")

#     # Flatten
#     points_flatten = points_flatten[mask_valid]
#     points_flatten[:,2] = points_flatten[:,2] - points_interpolated[:,2]

#     filtered_points = {dim: getattr(laz, dim)[mask_valid] for dim in laz.point_format.dimension_names}
#     header = laspy.LasHeader(point_format=laz.header.point_format, version=laz.header.version)
#     header.point_count = 0
#     new_las = laspy.LasData(header)


#     #   _Assign filtered and modified data
#     for dim, values in filtered_points.items():
#         setattr(new_las, dim, values)

#     setattr(new_las, 'x', points_flatten[:,0])
#     setattr(new_las, 'y', points_flatten[:,1])
#     setattr(new_las, 'z', points_flatten[:,2])

#     #   _Save new file
#     new_las.write(tile_new_original_src.split('.laz')[0] + "_flatten.laz")
#     if verbose:
#         print("Saved file: ", tile_new_original_src.split('.laz')[0] + "_flatten.laz")

#     # Resize original file
#     laz.points = laz.points[mask_valid]
#     laz.write(tile_new_original_src)
#     if verbose:
#         print("Saved file: ", tile_new_original_src)


# def flattening(src_tiles, src_new_tiles_loc, grid_size=10, verbose=True, do_keep_existing=False, verbose_full=False):
#     """
#     Applies the flattening process to all tiles in a directory using grid-based ground surface estimation.

#     Args:
#         - src_tiles (str): Path to the directory containing original tiles.
#         - src_new_tiles (str): Path to the directory where resized tiles will be saved.
#         - grid_size (int, optional): Size of the grid in meters for interpolation. Defaults to 10.
#         - verbose (bool, optional): Whether to show a general progress bar. Defaults to True.
#         - verbose_full (bool, optional): Whether to print detailed info per tile. Defaults to False.

#     Returns:
#         - None: Processes and saves flattened tiles into their respective folders.
#     """
    
#     os.makedirs(src_new_tiles_loc, exist_ok=True)

#     print("Starting flattening:")
#     list_tiles = [x for x in os.listdir(src_tiles) if x.endswith('.laz')]
#     for _, tile in tqdm(enumerate(list_tiles), total=len(list_tiles), desc="Processing", disable=verbose==False):
#         if verbose_full:
#             print("Flattening tile: ", tile)
#         if do_keep_existing and os.path.exists(os.path.join(src_new_tiles_loc, tile).split('.laz')[0] + "_flatten.laz"):
#             continue

#         flattening_tile(
#             tile_src=os.path.join(src_tiles, tile), 
#             tile_new_original_src=os.path.join(src_new_tiles_loc, tile),
#             grid_size=grid_size,
#             do_keep_existing=do_keep_existing,
#             verbose=verbose_full,
#             )

def flattening_tile(tile_src, tile_new_original_src, grid_size=10, method='cubic', do_save_floor=False, do_keep_existing=False, do_extrapolate_outside_hull=False, verbose=True):
    """
    Flattens a tile by interpolating the ground surface and subtracting it from the original elevation.

    Args:
        - tile_src (str): Path to the input tile in LAS/LAZ format.
        - tile_new_original_src (str): Path to save the resized original tile after filtering.
        - grid_size (int, optional): Size of the grid in meters for local interpolation. Defaults to 10.
        - verbose (bool, optional): Whether to display progress and debug information. Defaults to True.

    Returns:
        - None: Saves the floor and flattened versions of the tile and updates the original file.
    """
    if os.path.exists(tile_new_original_src) and do_keep_existing:
        if verbose:
            print(f"Skipping. {tile_new_original_src} exists already")
        return
    
    # Load file
    laz = laspy.read(tile_src)
    init_len = len(laz)
    if init_len == 0:
        return
    
    if verbose:
        print(f"Removing duplicates: From {init_len} to {len(laz)}")
    
    points = np.vstack((laz.x, laz.y, laz.z)).T
    points_flatten = copy.deepcopy(points)
    points_interpolated = copy.deepcopy(points)

    # Divide into tiles and find local minimums
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)

    x_bins = np.append(np.arange(x_min, x_max, grid_size), x_max)
    y_bins = np.append(np.arange(y_min, y_max, grid_size), y_max)

    grid = {i:{j:[] for j in range(y_bins.size - 1)} for i in range(x_bins.size -1)}
    for _, (px, py, pz) in tqdm(enumerate(points), total=len(points), desc="Creating grid", disable=verbose==False):
        xbin = np.clip(0, (px - x_min) // grid_size, x_bins.size - 2)
        ybin = np.clip(0, (py - y_min) // grid_size, y_bins.size - 2)
        try:
            grid[xbin][ybin].append((px, py, pz))
        except Exception as e:
            print("Problem with: ", tile_src)
            print(xbin)
            print(ybin)
            print(x_bins)
            print(y_bins)
            print(grid.keys())
            print(grid[0].keys())
            raise e


    # Create grid_min
    grid_used = np.zeros((x_bins.size - 1, y_bins.size - 1))
    lst_grid_min = []
    lst_grid_min_pos = []
    for x in grid.keys():
        for y in grid[x].keys():
            if np.array(grid[x][y]).shape[0] > 0:
                grid_used[x, y] = 1
                lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
                arg_min = np.argmin(np.array(grid[x][y])[:,2])
                lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2])

                # test if border
                if x == list(grid.keys())[0]:
                    lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
                    lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2] - [5, 0])
                if x == list(grid.keys())[-1]:
                    lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
                    lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2] + [5, 0])
                if y == list(grid[x].keys())[0]:
                    lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
                    lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2] - [0, 5])
                if y == list(grid[x].keys())[-1]:
                    lst_grid_min.append(np.min(np.array(grid[x][y])[:,2]))
                    lst_grid_min_pos.append(np.array(grid[x][y])[arg_min,0:2] + [0, 5])
            else:
                grid_used[x, y] = 0
    arr_grid_min_pos = np.vstack(lst_grid_min_pos)
    if verbose:
        print("Resulting grid:")
        print(arr_grid_min_pos.shape)
        print(grid_used)

    # Interpolate
    points_xy = np.array(points)[:,0:2]
    if method == 'cubic':
        interpolated_min_z = scipy.interpolate.griddata(arr_grid_min_pos, np.array(lst_grid_min), points_xy, method="cubic", fill_value=-1)
    elif method == 'quadric':
        rbf = scipy.interpolate.Rbf(arr_grid_min_pos[:,0], arr_grid_min_pos[:,1], np.array(lst_grid_min), function='multiquadric', smooth=5)
        interpolated_min_z = rbf(points_xy[:,0], points_xy[:,1])
    else:
        raise ValueError("Wrong argument for method!")

    # Fill NaNs with nearest neighbor interpolation
    if do_extrapolate_outside_hull:
        nan_mask = interpolated_min_z == -1
        x = np.array(points)[:,0]
        y = np.array(points)[:,1]

        if np.any(nan_mask):
            interpolated_min_z[nan_mask] = scipy.interpolate.griddata(arr_grid_min_pos, np.array(lst_grid_min), (x[nan_mask], y[nan_mask]), method='nearest')

    mask_valid = np.array([x != -1 for x in list(interpolated_min_z)])
    points_interpolated = points_interpolated[mask_valid]
    points_interpolated[:, 2] = interpolated_min_z[mask_valid]

    if verbose:
        print("Interpolation:")
        print(f"Original number of points: {points.shape[0]}")
        print(f"Interpollated number of points: {points_interpolated.shape[0]} ({int(points_interpolated.shape[0] / points.shape[0]*100)}%)")

    # save floor
    filtered_points = {dim: getattr(laz, dim)[mask_valid] for dim in laz.point_format.dimension_names}
    header = laspy.LasHeader(point_format=laz.header.point_format, version=laz.header.version)
    new_las = laspy.LasData(header)

    #   _Assign filtered and modified data
    for dim, values in filtered_points.items():
        setattr(new_las, dim, values)
    setattr(new_las, 'x', points_interpolated[:,0])
    setattr(new_las, 'y', points_interpolated[:,1])
    setattr(new_las, 'z', points_interpolated[:,2])

    #   _Save new file
    if do_save_floor:
        new_las.write(tile_new_original_src.split('.laz')[0] + "_floor.laz")
        if verbose:
            print("Saved file: ", tile_new_original_src.split('.laz')[0] + "_floor.laz")

    # Flatten
    points_flatten = points_flatten[mask_valid]
    points_flatten[:,2] = points_flatten[:,2] - points_interpolated[:,2]

    filtered_points = {dim: getattr(laz, dim)[mask_valid] for dim in laz.point_format.dimension_names}
    header = laspy.LasHeader(point_format=laz.header.point_format, version=laz.header.version)
    header.point_count = 0
    new_las = laspy.LasData(header)


    #   _Assign filtered and modified data
    for dim, values in filtered_points.items():
        setattr(new_las, dim, values)

    setattr(new_las, 'x', points_flatten[:,0])
    setattr(new_las, 'y', points_flatten[:,1])
    setattr(new_las, 'z', points_flatten[:,2])

    #   _Save new file
    new_las.write(tile_new_original_src.split('.laz')[0] + "_flatten.laz")
    if verbose:
        print("Saved file: ", tile_new_original_src.split('.laz')[0] + "_flatten.laz")

    # Resize original file
    laz.points = laz.points[mask_valid]
    laz.write(tile_new_original_src)
    if verbose:
        print("Saved file: ", tile_new_original_src)


def flattening(src_tiles, src_new_tiles, grid_size=10, verbose=True, method='cubic', do_save_floor=True, do_keep_existing=False, verbose_full=False):
    """
    Applies the flattening process to all tiles in a directory using grid-based ground surface estimation.

    Args:
        - src_tiles (str): Path to the directory containing original tiles.
        - src_new_tiles (str): Path to the directory where resized tiles will be saved.
        - grid_size (int, optional): Size of the grid in meters for interpolation. Defaults to 10.
        - verbose (bool, optional): Whether to show a general progress bar. Defaults to True.
        - verbose_full (bool, optional): Whether to print detailed info per tile. Defaults to False.

    Returns:
        - None: Processes and saves flattened tiles into their respective folders.
    """
    
    print("Starting flattening:")
    os.makedirs(src_new_tiles, exist_ok=True)
    list_tiles = [x for x in os.listdir(src_tiles) if x.endswith('.laz')]
    for _, tile in tqdm(enumerate(list_tiles), total=len(list_tiles), desc="Processing", disable=verbose==False):
        if verbose_full:
            print("Flattening tile: ", tile)
        if do_keep_existing and os.path.exists(os.path.join(src_new_tiles, tile).split('.laz')[0] + "_flatten.laz"):
            continue

        flattening_tile(
            tile_src=os.path.join(src_tiles, tile), 
            tile_new_original_src=os.path.join(src_new_tiles, tile),
            grid_size=grid_size,
            method=method, 
            do_save_floor=do_save_floor,
            do_keep_existing=do_keep_existing,
            verbose=verbose_full,
            )
        

def merge_laz(list_files, output_file):
    """
    Merge multiple LAS/LAZ files into one using laspy.
    Preserves all dimensions including extra_dims like 'id_point'.
    """
    # Read the header from the first file
    first_las = laspy.read(list_files[0])
    header = laspy.LasHeader(point_format=first_las.header.point_format,
                              version=first_las.header.version)

    all_arrays = []
    for _, f in tqdm(enumerate(list_files), total=len(list_files)):
        las = laspy.read(f)
        all_arrays.append(las.points.array)  # extract structured array
        # print(f"Read {f} ({len(las)} pts)")

    # Concatenate structured arrays
    merged_array = np.concatenate(all_arrays)

    # Create new LasData with header
    out = laspy.LasData(header)

    # Wrap the concatenated array as ScaleAwarePointRecord and assign
    out.points = laspy.ScaleAwarePointRecord(
        merged_array,
        point_format=header.point_format,
        scales=header.scales,
        offsets=header.offsets
    )

    # Save
    out.write(output_file)
    print(f"Merged file saved to {output_file}")


def preprocess():
    # loading
    start_preprocess_time = time()
    laz_original = laspy.read(SRC_INPUT)

    # prepare paths
    src_with_id = SRC_INPUT.split('.laz')[0] + "_with_id.laz"
    src_folder_tiles_wo_overlap = os.path.join(os.path.dirname(src_with_id), os.path.basename(src_with_id).split('.laz')[0] + "_tiles_no_overlap")
    src_folder_tiles_w_overlap = os.path.join(os.path.dirname(src_with_id), os.path.basename(src_with_id).split('.laz')[0] + "_tiles_overlap")
    src_folder_flatten_tiles_w_overlap = os.path.join(os.path.dirname(src_with_id), os.path.basename(src_with_id).split('.laz')[0] + "_flatten")
    src_flatten_file = os.path.join(SRC_INPUT.split('.laz')[0] + "_flatten.laz")
    src_folder_stripes = os.path.join(os.path.dirname(src_flatten_file), os.path.basename(src_flatten_file).split('.laz')[0] + f"_stripes_{STRIPE_WIDTH}_m")

    # adding an id to the points
    id_point = np.arange(len(laz_original))
    laz_original.add_extra_dim(laspy.ExtraBytesParams('id_point', type="uint32"))
    laz_original.id_point = id_point
    laz_original.write(src_with_id)

    # tiles without overlap
    tilling(
        src_input=src_with_id, 
        src_target=src_folder_tiles_wo_overlap, 
        tile_size=TILE_SIZE,
        overlap=0,
        shift=SHIFT)

    # tiles with overlap
    tilling(
        src_input=src_with_id, 
        src_target=src_folder_tiles_w_overlap, 
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        shift=SHIFT)

    # Flattening of tiles with overlap
    flattening(
        src_tiles=src_folder_tiles_w_overlap,
        src_new_tiles=src_folder_flatten_tiles_w_overlap,
        grid_size=GRID_SIZE,
        method=METHOD,
        do_save_floor=do_save_floor,
        do_keep_existing=do_keep_existing_flatten,
        verbose=True,
        verbose_full=False,
    )

    # Creating flatten tiles w/o overlap and merging
    list_flatten_to_merge = []
    print("Creating flaten tiles without overlap")
    list_tiles = [x for x in os.listdir(src_folder_tiles_wo_overlap) if x.endswith('.laz')]
    for _, tile in tqdm(enumerate(list_tiles), total=len(list_tiles)):
        assert os.path.exists(os.path.join(src_folder_tiles_wo_overlap, tile))

        laz_with_ov = laspy.read(os.path.join(src_folder_flatten_tiles_w_overlap, tile))
        laz_without_ov = laspy.read(os.path.join(src_folder_tiles_wo_overlap, tile))
        laz_flatten_with_ov = laspy.read(os.path.join(src_folder_flatten_tiles_w_overlap, tile.split('.laz')[0] + "_flatten.laz"))

        mask = np.isin(laz_with_ov.id_point, laz_without_ov.id_point)

        laz_flatten_wo_ov = laz_flatten_with_ov[mask]
        laz_flatten_wo_ov.write(os.path.join(src_folder_flatten_tiles_w_overlap, tile.split('.laz')[0] + "_flatten_no_ov.laz"))
        list_flatten_to_merge.append(os.path.join(src_folder_flatten_tiles_w_overlap, tile.split('.laz')[0] + "_flatten_no_ov.laz"))

    # Merging
    print("Merging all flatten tiles together (might take a few minutes)")
    src_flatten_file = os.path.join(SRC_INPUT.split('.laz')[0] + "_flatten.laz")
    merge_laz(list_flatten_to_merge, src_flatten_file)

    # Generate stripes from the merged flatten
    x_span = laz_original.x.max() - laz_original.x.min()
    y_span = laz_original.y.max() - laz_original.y.min()
    dims = [x_span, y_span]

    dims[np.argmin([x_span, y_span])] = STRIPE_WIDTH

    print("Creation of stripes:")
    stripes_file(src_flatten_file, src_folder_stripes, dims, do_keep_existing=True)

    delta_time_loop = time() - start_preprocess_time
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"==== Preprocess done in {hours}:{min}:{sec} ====")


def postprocessing():
    print("postprocess!")


def main():
    if MODE == "PRE":
        preprocess()
    elif MODE == "POST":
        postprocessing()
    else:
        raise ValueError("Wrong INPUT for variable MODE!")


if __name__ == "__main__":
    main()
