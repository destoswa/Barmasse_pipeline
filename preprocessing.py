import os
import shutil
import traceback
import numpy as np
import laspy
from scipy.interpolate import griddata, RBFInterpolator
import copy
from tqdm import tqdm
from time import time
from itertools import product
from omegaconf import OmegaConf
from format_conversions import convert_all_in_folder


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

    combinations = list(product(range(x_steps), range(y_steps)))
    
    for _, (ix, iy) in tqdm(enumerate(combinations), total=len(combinations), desc="Creation of tiles"):
        x0 = x_min + ix * tile_size
        x1 = x_min + (ix + 1) * tile_size
        y0 = y_min + iy * tile_size
        y1 = y_min + (iy + 1) * tile_size

        mask_wo_overlap = (
            (las.x >= x0) & (las.x <= x1) &
            (las.y >= y0) & (las.y <= y1)
        )
        x0 -= overlap
        y0 -= overlap
        x1 += overlap
        y1 += overlap

        mask_with_overlap = (
            (las.x >= x0) & (las.x <= x1) &
            (las.y >= y0) & (las.y <= y1)
        )
        if np.sum(mask_wo_overlap) < 100:
            continue

        # ✅ Create new file with proper header scale/offset
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        header.offsets = las.header.offsets
        header.scales = las.header.scales

        # ✅ Copy CRS if any
        if hasattr(las.header, "epsg") and las.header.epsg is not None:
            header.epsg = las.header.epsg

        tile = laspy.LasData(header)
        tile.points = las.points[mask_with_overlap]

        tile_filename = os.path.join(
            src_target,
            f"{os.path.splitext(os.path.basename(src_input))[0]}_tile_{ix}_{iy}.laz"
        )
        tile.write(tile_filename)


def stripes_file(src_input_file, src_output, dims, verbose=True):
    """
    Crops a LAS/LAZ file into tiles using laspy directly (preserves id_point and coordinates).
    """
    [tile_size_x, tile_size_y] = dims
    las = laspy.read(src_input_file)
    os.makedirs(src_output, exist_ok=True)

    xmin, xmax, ymin, ymax = las.x.min(), las.x.max(), las.y.min(), las.y.max()

    # print(xmin)
    # print(xmax)
    # print(tile_size_x)
    # print('...')
    x_edges = np.arange(xmin, xmax, tile_size_x)
    y_edges = np.arange(ymin, ymax, tile_size_y)

    # print(x_edges.shape)
    # print(y_edges.shape)

    combinations = list(product(x_edges, y_edges))
    # for (x0, y0) in combinations:
    num_skipped = 0
    for id_stripe, (x0, y0) in tqdm(enumerate(combinations), total=len(combinations), desc="Processing"):
        x1 = x0 + tile_size_x
        y1 = y0 + tile_size_y

        mask = (
            (las.x >= x0) & (las.x <= x1) &
            (las.y >= y0) & (las.y <= y1)
        )
        if not np.any(mask):
            num_skipped += 1
            continue

        # Create new file with proper header scale/offset
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        header.offsets = las.header.offsets
        header.scales = las.header.scales

        # Copy CRS if any
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


def flattening_tile(tile_src, tile_new_original_src, grid_size=10, method='cubic', epsilon=0.2, do_save_floor=False, do_keep_existing=False, verbose=True):
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
        interpolated_min_z = griddata(arr_grid_min_pos, np.array(lst_grid_min), points_xy, method="cubic")
    elif method == 'multiquadric':
        interpolated_min_z = RBFInterpolator(arr_grid_min_pos, np.array(lst_grid_min), kernel='multiquadric', epsilon=epsilon)(points_xy)
    elif method == 'invmultiquadric':
        interpolated_min_z = RBFInterpolator(arr_grid_min_pos, np.array(lst_grid_min), kernel='inverse_multiquadric', epsilon=epsilon)(points_xy)
    else:
        raise ValueError("Wrong argument for method!")

    mask_valid = np.array([x != -1 for x in list(interpolated_min_z)])
    points_interpolated = points_interpolated[mask_valid]
    points_interpolated[:, 2] = interpolated_min_z[mask_valid]

    if verbose:
        print("Interpolation:")
        print(f"Original number of points: {points.shape[0]}")
        print(f"Interpollated number of points: {points_interpolated.shape[0]} ({int(points_interpolated.shape[0] / points.shape[0]*100)}%)")

    # Resize original file
    laz.points = laz.points[mask_valid]
    laz.write(tile_new_original_src)
    if verbose:
        print("Saved file: ", tile_new_original_src)

    # Floor
    setattr(laz, 'xyz', points_interpolated)

    #   _Save new file
    if do_save_floor:
        laz.write(tile_new_original_src.split('.laz')[0] + "_floor.laz")
        if verbose:
            print("Saved file: ", tile_new_original_src.split('.laz')[0] + "_floor.laz")

    # Flatten
    points_flatten = points_flatten[mask_valid]
    points_flatten[:,2] = points_flatten[:,2] - points_interpolated[:,2]

    setattr(laz, 'xyz', points_flatten)

    #   _Save new file
    laz.write(tile_new_original_src.split('.laz')[0] + "_flatten.laz")
    if verbose:
        print("Saved file: ", tile_new_original_src.split('.laz')[0] + "_flatten.laz")


def flattening(src_tiles, src_new_tiles, grid_size=10, verbose=True, method='cubic', epsilon=0.2, do_save_floor=True, do_skip_existing=False, verbose_full=False):
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

    os.makedirs(src_new_tiles, exist_ok=True)

    list_tiles = [x for x in os.listdir(src_tiles) if x.endswith('.laz')]
    for _, tile in tqdm(enumerate(list_tiles), total=len(list_tiles), desc="Processing", disable=verbose==False):
        if verbose_full:
            print("Flattening tile: ", tile)

        if do_skip_existing == True and os.path.exists(os.path.join(src_new_tiles, tile).split('.laz')[0] + "_flatten.laz"):
            if verbose_full:
                print(f"Skipping. {tile} exists already")
            continue

        flattening_tile(
            tile_src=os.path.join(src_tiles, tile), 
            tile_new_original_src=os.path.join(src_new_tiles, tile),
            grid_size=grid_size,
            method=method, 
            epsilon=epsilon,
            do_save_floor=do_save_floor,
            verbose=verbose_full,
            )
        

def merge_laz(list_files, output_file):
    # Load first file (keeps header intact)
    out = laspy.read(list_files[0])

    # Append points from others
    for f in list_files[1:]:
        las = laspy.read(f)
        out.points = laspy.ScaleAwarePointRecord(
            np.concatenate([out.points.array, las.points.array]),
            point_format=out.header.point_format,
            scales=out.header.scales,
            offsets=out.header.offsets
        )

    # Update bounding box (important)
    xs = out.x
    ys = out.y
    zs = out.z

    out.header.min = [float(xs.min()), float(ys.min()), float(zs.min())]
    out.header.max = [float(xs.max()), float(ys.max()), float(zs.max())]
    
    # Save
    out.write(output_file)
    print(f"Merged {len(list_files)} files → {output_file}")


def hash_coords(x, y, z, rounding=3):
    """Create a unique integer hash for each rounded coordinate triple."""
    return np.round(x, rounding) * 1e12 + np.round(y, rounding) * 1e6 + np.round(z, rounding)


def main(args):
    """
    Preprocessing of the pointcloud into flatten stripes.
    The process is subdivided into the following steps:
    	- splitting as tiles wo overlap
        - splitting as tiles w overlap
        - flattening of tiles w overlap
        - creation of flatten tiles wo overlap
        - creation of floor tiles wo overlap
        - merging flatten tiles wo overlap into big flatten pointcloud
        - merging floor tiles wo overlap into big floor
        - striping of big flatten pc
        - striping of big floor
    """

    # hyperparameters
    SRC_INPUT = r'%s' %args.src_point_cloud
    INPUT_TYPE = SRC_INPUT.split('.')[-1].lower()
    TILE_SIZE = int(args.tile_size)
    OVERLAP = int(args.overlap)
    GRID_SIZE = int(args.grid_size)
    STRIPE_WIDTH = int(args.stripe_width)
    STRIPE_DIM = args.stripe_dim
    METHOD = args.method
    METHOD_EPSILON = float(args.method_epsilon)
    OUTPUT_TYPE = args.output_type
    DO_SKIP_EXISTING_FLATTEN = args.do_skip_existing_flatten
    SKIP_TO_STEP = int(args.skip_to_step)

    # prepare paths
    src_preprocess_dir = os.path.join(os.path.dirname(SRC_INPUT), 'preprocessing_files')
    src_folder_tiles_wo_overlap = os.path.join(src_preprocess_dir, os.path.basename(SRC_INPUT).split(f'.{INPUT_TYPE}')[0] + "_tiles_no_overlap")
    src_folder_tiles_w_overlap = os.path.join(src_preprocess_dir, os.path.basename(SRC_INPUT).split(f'.{INPUT_TYPE}')[0] + "_tiles_overlap")
    src_folder_flatten_tiles = os.path.join(src_preprocess_dir, os.path.basename(SRC_INPUT).split(f'.{INPUT_TYPE}')[0] + "_flatten")
    src_flatten_file = os.path.join(src_preprocess_dir, os.path.basename(SRC_INPUT).split(f'.{INPUT_TYPE}')[0] + "_flatten.laz")
    src_floor_file = os.path.join(src_preprocess_dir, os.path.basename(SRC_INPUT).split(f'.{INPUT_TYPE}')[0] + "_floor.laz")
    src_folder_stripes_flatten = os.path.join(os.path.dirname(src_flatten_file), os.path.basename(SRC_INPUT).split(f'.{INPUT_TYPE}')[0] + f"_stripes_flatten_{STRIPE_WIDTH}_m")
    src_folder_stripes_floor = os.path.join(os.path.dirname(src_flatten_file), os.path.basename(SRC_INPUT).split(f'.{INPUT_TYPE}')[0] + f"_stripes_floor_{STRIPE_WIDTH}_m")
    src_folder_stripes_original = os.path.join(os.path.dirname(src_flatten_file), os.path.basename(SRC_INPUT).split(f'.{INPUT_TYPE}')[0] + f"_stripes_original_{STRIPE_WIDTH}_m")
    src_to_process_dir = os.path.join(os.path.dirname(SRC_INPUT), "to_process")
    
    # 1 - splitting as tiles wo overlap
    if SKIP_TO_STEP <= 1:
        print("\nStep 1/11 - Tilling without overlap")
        tilling(
            src_input=SRC_INPUT, 
            src_target=src_folder_tiles_wo_overlap, 
            tile_size=TILE_SIZE,
            overlap=0)
    
    # 2 - splitting as tiles w overlap
    if SKIP_TO_STEP <= 2:
        print("\n---\n\nStep 2/11 - Tilling with overlap")
        tilling(
            src_input=SRC_INPUT, 
            src_target=src_folder_tiles_w_overlap, 
            tile_size=TILE_SIZE,
            overlap=OVERLAP)
    
	# 3 - flattening of tiles w overlap
    if SKIP_TO_STEP <= 3:
        print("\n---\n\nStep 3/11 - Flattening of tiles")
        flattening(
            src_tiles=src_folder_tiles_w_overlap,
            src_new_tiles=src_folder_flatten_tiles,
            grid_size=GRID_SIZE,
            method=METHOD,
            epsilon=METHOD_EPSILON,
            do_skip_existing=DO_SKIP_EXISTING_FLATTEN,
            verbose=True,
            verbose_full=False,
        )
	
    # 4 - creation of flatten tiles wo overlap
    if SKIP_TO_STEP <= 4:
        print("\n---\n\nStep 4/11 - Creating flatten tiles without overlap")
        list_flatten_to_merge = []
        list_tiles = [x for x in os.listdir(src_folder_tiles_wo_overlap) if x.endswith('.laz')]
        for _, tile in tqdm(enumerate(list_tiles), total=len(list_tiles), desc="Processing"):
            # print(tile)
            assert os.path.exists(os.path.join(src_folder_tiles_wo_overlap, tile))
            try:
                laz_with_ov = laspy.read(os.path.join(src_folder_flatten_tiles, tile))
            except:
                print("Error with : ", os.path.join(src_folder_flatten_tiles, tile))
                continue
            laz_without_ov = laspy.read(os.path.join(src_folder_tiles_wo_overlap, tile))
            laz_flatten_with_ov = laspy.read(os.path.join(src_folder_flatten_tiles, tile.split('.laz')[0] + "_flatten.laz"))

            hash_without_ov = hash_coords(laz_without_ov.x, laz_without_ov.y, laz_without_ov.z, 3)
            hash_with_ov = hash_coords(laz_with_ov.x, laz_with_ov.y, laz_with_ov.z, 3)

            mask = np.isin(hash_with_ov, hash_without_ov)

            laz_flatten_wo_ov = laz_flatten_with_ov[mask]

            src_flatten_wo_ov = os.path.join(src_folder_flatten_tiles, tile.split('.laz')[0] + "_flatten_wo_ov.laz")
            laz_flatten_wo_ov.write(src_flatten_wo_ov)
            list_flatten_to_merge.append(src_flatten_wo_ov)

	# 5 - merging flatten tiles wo overlap into big flatten pointcloud
    if SKIP_TO_STEP <= 5:
        print("\n---\n\nStep 5/11 - Merging all flatten tiles together (might take a few minutes)")
        merge_laz(list_flatten_to_merge, src_flatten_file)

	# 6 - creation of floor tiles wo overlap
    if SKIP_TO_STEP <= 6:
        print("\n---\n\nStep 6/11 - Creating floor tiles without overlap")
        list_floor_to_merge = []
        list_tiles = [x for x in os.listdir(src_folder_tiles_wo_overlap) if x.endswith('.laz')]
        for _, tile in tqdm(enumerate(list_tiles), total=len(list_tiles), desc="Processing"):
            # print(tile)
            assert os.path.exists(os.path.join(src_folder_tiles_wo_overlap, tile))
            try:
                laz_with_ov = laspy.read(os.path.join(src_folder_flatten_tiles, tile))
            except:
                print("Error with : ", os.path.join(src_folder_flatten_tiles, tile))
                continue

            laz_without_ov = laspy.read(os.path.join(src_folder_tiles_wo_overlap, tile))
            laz_floor_with_ov = laspy.read(os.path.join(src_folder_flatten_tiles, tile.split('.laz')[0] + "_floor.laz"))

            hash_without_ov = hash_coords(laz_without_ov.x, laz_without_ov.y, laz_without_ov.z, 3)
            hash_with_ov = hash_coords(laz_with_ov.x, laz_with_ov.y, laz_with_ov.z, 3)

            mask = np.isin(hash_with_ov, hash_without_ov)

            laz_floor_wo_ov = laz_floor_with_ov[mask]
            src_floor_wo_ov = os.path.join(src_folder_flatten_tiles, tile.split('.laz')[0] + "_floor_wo_ov.laz")
            laz_floor_wo_ov.write(src_floor_wo_ov)
            list_floor_to_merge.append(src_floor_wo_ov)

	# 7 - merging floor tiles wo overlap into big floor
    if SKIP_TO_STEP <= 7:
        print("\n---\n\nStep 7/11 - Merging all flatten tiles together (might take a few minutes)")
        merge_laz(list_floor_to_merge, src_floor_file)
    
	# 8 - striping of big flatten pc
    if SKIP_TO_STEP <= 8:
        print("\n---\n\nStep 8/11 - Creation of flatten stripes:")
        laz_flatten = laspy.read(src_flatten_file)
        x_span = laz_flatten.x.max() - laz_flatten.x.min()
        y_span = laz_flatten.y.max() - laz_flatten.y.min()
        if STRIPE_DIM == 'x':
            dims = [STRIPE_WIDTH, y_span]
        elif STRIPE_DIM == 'y':
            dims = [x_span, STRIPE_WIDTH]
        else:
            raise ValueError("Wrong input for 'stripe dim' parameter! (need to be 'x' or 'y')")
        stripes_file(src_flatten_file, src_folder_stripes_flatten, dims)

	# 9 - striping of big floor
    if SKIP_TO_STEP <= 9:
        print("---\nStep 9/11 - Creation of floor stripes:")
        laz_floor = laspy.read(src_floor_file)
        x_span = laz_floor.x.max() - laz_floor.x.min()
        y_span = laz_floor.y.max() - laz_floor.y.min()
        if STRIPE_DIM == 'x':
            dims = [STRIPE_WIDTH, y_span]
        elif STRIPE_DIM == 'y':
            dims = [x_span, STRIPE_WIDTH]
        else:
            raise ValueError("Wrong input for 'stripe dim' parameter! (need to be 'x' or 'y')")

        stripes_file(src_floor_file, src_folder_stripes_floor, dims)

	# 10 - striping of big original
    if SKIP_TO_STEP <= 10:
        print("---\nStep 10/11 - Creation of original stripes:")
        laz_original = laspy.read(SRC_INPUT)
        x_span = laz_original.x.max() - laz_original.x.min()
        y_span = laz_original.y.max() - laz_original.y.min()
        if STRIPE_DIM == 'x':
            dims = [STRIPE_WIDTH, y_span]
        elif STRIPE_DIM == 'y':
            dims = [x_span, STRIPE_WIDTH]
        else:
            raise ValueError("Wrong input for 'stripe dim' parameter! (need to be 'x' or 'y')")

        stripes_file(SRC_INPUT, src_folder_stripes_original, dims)

    # 11 - striping of big original
    if SKIP_TO_STEP <= 11:
        print("---\nStep 11/11 - Preparation of stripes for manual cleaning (in folder 'to_process'):")
        # Save offset if txt (processing pc with Polyworks)
        if OUTPUT_TYPE == 'txt':
            try: laz_original
            except: laz_original = laspy.read(SRC_INPUT)
            # laz_original = laspy.read(SRC_INPUT) if laz_original else laz_original
            offsets = [int(x) for x in laz_original.header.offsets]
            src_offsets = os.path.join(os.path.dirname(SRC_INPUT), 'offsets.txt')
            with open(src_offsets, 'w') as f:
                f.write(f"{offsets[0]},{offsets[1]},{offsets[2]}")
        
        # Copy results into a folder for cleaning
        if OUTPUT_TYPE == INPUT_TYPE:
            shutil.copytree(src_folder_stripes_flatten, src_to_process_dir, dirs_exist_ok=True)
        else:
            try: offsets
            except: offsets = [0,0,0]
            convert_all_in_folder(src_folder_stripes_flatten, src_to_process_dir, INPUT_TYPE, OUTPUT_TYPE, offsets=offsets, verbose=True)
    



if __name__ == "__main__":
    conf = OmegaConf.load('configs.yaml')

    print(f"Preprocessing file: \n\t{conf.preprocessing.src_point_cloud} \nwith the following configuration:")
    for key, val in conf.preprocessing.items():
        if key != 'src_point_cloud':
            print(f"\t - {key}: {val}")
    print("-" * 20)

    start_preprocess_time = time()

    try:
        main(conf.preprocessing)
    except Exception as e:
        print(traceback.format_exc())
        input("\nPress enter to continue...")
        quit()

    # Showing duration of process
    delta_time_loop = time() - start_preprocess_time
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"\n==== Preprocessing done in {hours}:{min}:{sec} ====\n")

    input("Press enter to continue...")
