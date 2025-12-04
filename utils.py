import os
import numpy as np
import laspy
import shutil
from scipy.interpolate import griddata, RBFInterpolator, Rbf
from sklearn.neighbors import NearestNeighbors 
import copy
from tqdm import tqdm
from itertools import product
from multiprocessing import Pool, cpu_count
from itertools import product
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def tilling(src_input, src_target, tile_size, overlap=0, shift=0, verbose=True):
    """
    Crops a LAS/LAZ file into tiles using laspy directly (preserves id_point and coordinates).
    """
    if os.path.exists(src_target):
        shutil.rmtree(src_target)
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
    if os.path.exists(src_output):
        shutil.rmtree(src_output)
    os.makedirs(src_output, exist_ok=True)
    
    [tile_size_x, tile_size_y] = dims
    las = laspy.read(src_input_file)

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

    # Number of bins
    num_x = x_bins.size - 1
    num_y = y_bins.size - 1

    # Total number of grid cells
    num_bins = num_x * num_y

    # Create fast flat list-of-lists for accumulation
    cells = [[] for _ in range(num_bins)]

    # ----- FAST BINNING LOOP -----
    for px, py, pz in points:
        # Compute bin indices
        xbin = int((px - x_min) // grid_size)
        ybin = int((py - y_min) // grid_size)

        # Clamp values
        if xbin < 0:
            xbin = 0
        elif xbin >= num_x:
            xbin = num_x - 1

        if ybin < 0:
            ybin = 0
        elif ybin >= num_y:
            ybin = num_y - 1

        # Flat index
        fid = xbin * num_y + ybin

        # Add point
        cells[fid].append((px, py, pz))


    grid_used = np.zeros((num_x, num_y), dtype=int)
    lst_grid_min = []
    lst_grid_min_pos = []

    for x in range(num_x):
        for y in range(num_y):
            fid = x * num_y + y
            cell_points = np.asarray(cells[fid])
            if cell_points.size == 0:
                grid_used[x, y] = 0
                continue

            grid_used[x, y] = 1
            z_values = cell_points[:, 2]
            arg_min = np.argmin(z_values)
            z_min = z_values[arg_min]
            pos_min = cell_points[arg_min, 0:2]

            # Append the minimum point
            lst_grid_min.append(z_min)
            lst_grid_min_pos.append(pos_min)

            # Borders
            if x == 0:
                lst_grid_min.append(z_min)
                lst_grid_min_pos.append(pos_min + np.array([-5, 0]))
            if x == num_x - 1:
                lst_grid_min.append(z_min)
                lst_grid_min_pos.append(pos_min + np.array([5, 0]))
            if y == 0:
                lst_grid_min.append(z_min)
                lst_grid_min_pos.append(pos_min + np.array([0, -5]))
            if y == num_y - 1:
                lst_grid_min.append(z_min)
                lst_grid_min_pos.append(pos_min + np.array([0, 5]))

    arr_grid_min_pos = np.vstack(lst_grid_min_pos)

    if verbose:
        print("Resulting grid:")
        print(arr_grid_min_pos.shape)
        print(grid_used)

    # temp_time = time()
    if epsilon is 'auto':
        if method == 'multiquadric':
            # default epsilon is the "the average distance between nodes" based on a bounding hypercube
            ximax = np.amax(arr_grid_min_pos, axis=0)
            ximin = np.amin(arr_grid_min_pos, axis=0)
            N = arr_grid_min_pos.shape[-1]
            edges = ximax - ximin
            edges = edges[np.nonzero(edges)]
            epsilon = np.power(np.prod(edges)/N, 1.0/edges.size)
        elif method == 'invmultiquadric':
            nbrs = NearestNeighbors(n_neighbors=2).fit(points)
            distances, _ = nbrs.kneighbors(points)
            d = np.median(distances[:, 1])  # median NN distance
            epsilon = 1.0 / d
        else:
            pass
        
    # Interpolate
    points_xy = np.array(points)[:,0:2]
    if method == 'cubic':
        interpolated_min_z = griddata(arr_grid_min_pos, np.array(lst_grid_min), points_xy, method="cubic")
    elif method == 'multiquadric':
        # rbf = Rbf(arr_grid_min_pos[:,0], arr_grid_min_pos[:,1], np.array(lst_grid_min), function='multiquadric', smooth=5)
        # interpolated_min_z = rbf(points_xy[:,0], points_xy[:,1])
        interpolated_min_z = RBFInterpolator(arr_grid_min_pos, np.array(lst_grid_min), kernel='multiquadric', epsilon=epsilon)(points_xy)
    elif method == 'invmultiquadric':
        interpolated_min_z = RBFInterpolator(arr_grid_min_pos, np.array(lst_grid_min), kernel='inverse_multiquadric', epsilon=epsilon)(points_xy)
    else:
        raise ValueError("Wrong argument for method!")
    # print("Duration 3: ", time() - temp_time)

    # temp_time = time()
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
    # print("Duration 4: ", time() - temp_time)

    return epsilon


def _flatten_tile_worker(args):
    """
    Worker function for processing a single tile in parallel.
    """
    tile, src_tiles, src_new_tiles, grid_size, method, epsilon, do_save_floor, do_skip_existing, verbose_full = args

    tile_src_path = os.path.join(src_tiles, tile)
    tile_new_path = os.path.join(src_new_tiles, tile).split('.laz')[0] + "_flatten.laz"

    # Skip existing if needed
    if do_skip_existing and os.path.exists(tile_new_path):
        if verbose_full:
            print(f"Skipping {tile} (already exists)")
        return tile  # can return skipped tile info

    if verbose_full:
        print(f"Flattening tile: {tile}")

    epsilon_out = flattening_tile(
        tile_src=tile_src_path,
        tile_new_original_src=os.path.join(src_new_tiles, tile),
        grid_size=grid_size,
        method=method,
        epsilon=epsilon,
        do_save_floor=do_save_floor,
        verbose=verbose_full,
    )
    return epsilon_out  # optionally return tile processed


def flattening(src_tiles, src_new_tiles, grid_size=10, method='cubic',
                        epsilon=0.2, do_save_floor=True, do_skip_existing=False, 
                        use_multiprocessing=False, n_processes=None, verbose=True, verbose_full=False):
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

    """
    Flatten all tiles in parallel using multiprocessing.
    """
    os.makedirs(src_new_tiles, exist_ok=True)

    list_tiles = [x for x in os.listdir(src_tiles) if x.endswith('.laz')]

    do_generate_custom_eps = epsilon == None
    lst_cust_eps = []

    if use_multiprocessing:
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)  # leave one CPU free

        # Prepare args for each worker
        worker_args = [(tile, src_tiles, src_new_tiles, grid_size, method, epsilon,
                        do_save_floor, do_skip_existing, verbose_full)
                    for tile in list_tiles]

        # Use a pool
        with Pool(processes=n_processes) as pool:
            # Using tqdm for progress bar
            for eps_out in tqdm(pool.imap_unordered(_flatten_tile_worker, worker_args),
                        total=len(worker_args), desc="Processing", disable=not verbose):
                lst_cust_eps.append(eps_out)
    else:
        for _, tile in tqdm(enumerate(list_tiles), total=len(list_tiles), desc="Processing", disable=verbose==False):
            if verbose_full:
                print("Flattening tile: ", tile)

            if do_skip_existing == True and os.path.exists(os.path.join(src_new_tiles, tile).split('.laz')[0] + "_flatten.laz"):
                if verbose_full:
                    print(f"Skipping. {tile} exists already")
                continue

            eps_out = flattening_tile(
                tile_src=os.path.join(src_tiles, tile), 
                tile_new_original_src=os.path.join(src_new_tiles, tile),
                grid_size=grid_size,
                method=method, 
                epsilon=epsilon,
                do_save_floor=do_save_floor,
                verbose=verbose_full,
                )
            lst_cust_eps.append(eps_out)
    
    if do_generate_custom_eps and len(lst_cust_eps) > 0:
        print("\t- Epsilon was computed automatically with a mean value of ", np.round(np.mean(lst_cust_eps), 3))
        

def merge_laz(list_files, output_file, verbose=False):
    # Load first file (keeps header intact)
    out = None

    # Append points from others
    for num_f, f in tqdm(enumerate(list_files), total=len(list_files), desc="Processing"):
        if num_f == 0:
            out = laspy.read(list_files[0])
        else:
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
    if verbose:
        print(f"Merged {len(list_files)} files → {output_file}")


def hash_coords(x, y, z, rounding=3):
    """Create a unique integer hash for each rounded coordinate triple."""
    return np.round(x, rounding) * 1e12 + np.round(y, rounding) * 1e6 + np.round(z, rounding)


def add_matching_z(xy_floor, z_floor, xyz_flatten, precision=3):
    """
    Add Z values based on XY matching with configurable rounding precision.

    xy_floor:    (M, 2) reference XY
    z_floor:     (M,)   reference Z
    xyz_flatten: (N, 3) points to update
    precision:   number of decimals used for matching
    """

    # Round all xy to given precision (ensures consistent matching)
    xy_floor_rounded = np.round(xy_floor, precision)
    xy_flatten_rounded = np.round(xyz_flatten[:, :2], precision)

    # Convert to tuples for hashing
    xy_ref_tuples = [tuple(row) for row in xy_floor_rounded]
    xy_query_tuples = [tuple(row) for row in xy_flatten_rounded]

    # Build dictionary: (x, y) → z
    mapping = dict(zip(xy_ref_tuples, z_floor))

    # Fast lookup
    matched_z = np.array([mapping.get(t, 0) for t in xy_query_tuples])

    # Add to z column
    xyz_flatten[:, 2] += matched_z

    return xyz_flatten


def matching_mask(xyz_original, xyz_clean, precision=3):
    """
    Return a boolean mask for xy_flatten indicating which points
    have a matching XY coordinate in xy_floor (after rounding).

    xy_floor:   (M, 2)
    xy_flatten: (N, 2)
    precision:  rounding precision for coordinate matching
    """

    # Round both arrays
    xyz_original_rounded = np.round(xyz_original, precision)
    xyz_clean_rounded = np.round(xyz_clean, precision)

    # Convert xy_flatten to a set for fast lookup
    clean_set = set(map(tuple, xyz_clean_rounded))

    # Build mask for floor entries
    mask = np.array([tuple(row) in clean_set for row in xyz_original_rounded])

    return mask
