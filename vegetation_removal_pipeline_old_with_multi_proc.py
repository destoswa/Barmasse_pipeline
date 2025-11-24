import os
import numpy as np
import laspy
import json
import scipy
import copy
from tqdm import tqdm
from time import time
import itertools
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import pyproj


# ===== INPUTS =====
SRC_INPUT = r"D:\GitHubProjects\Terranum_repo\TreeSegmentation\data\Barmasse\Barmasse_2025\Barmasse_2025_AllScans_Raw_Sub2cm_aligned.laz"
TILE_SIZE = 500
OVERLAP = 50
GRID_SIZE = 10
STRIPE_WIDTH = 5
# ==================





def stripes_file(src_input_file, src_output, dims):
    [tile_size_x, tile_size_y] = dims
    laz = laspy.read(src_input_file)
    # output_folder = os.path.join(src_output, f"{os.path.basename(src_input_file).split('.laz')[0]}_stripes_{tile_size_x}_{tile_size_y}")
    os.makedirs(src_output, exist_ok=True)

    xmin, xmax, ymin, ymax = np.min(laz.x), np.max(laz.x), np.min(laz.y), np.max(laz.y)

    x_edges = np.arange(xmin, xmax, tile_size_x)
    y_edges = np.arange(ymin, ymax, tile_size_y)


    for i, x0 in enumerate(x_edges):
        print(f"Column {i+1} / {len(x_edges)}:")
        for j, y0 in tqdm(enumerate(y_edges), total=len(y_edges)):
            x1 = x0 + tile_size_x
            y1 = y0 + tile_size_y
            bounds = f"([{x0},{x1}],[{y0},{y1}])"
            output = os.path.join(src_output,f"{os.path.basename(src_input_file).split('.laz')[0]}_stripe_{i}_{j}.laz")
            pipeline_json = {
                "pipeline": [
                    src_input_file,
                    {"type": "filters.crop", "bounds": bounds},
                    {"type": "writers.las", "filename": output, 'extra_dims': 'all'}
                ]
            }

            pipeline = pdal.Pipeline(json.dumps(pipeline_json))
            pipeline.execute()


def stripes_file_multi_intern(src_input_file, stripe_width):
    src_output = os.path.join(os.path.dirname(src_input_file), os.path.basename(src_input_file).split('.laz')[0] + '_stripes')
    # [tile_size_x, tile_size_y] = dims
    laz = laspy.read(src_input_file)
    # output_folder = os.path.join(src_output, f"{os.path.basename(src_input_file).split('.laz')[0]}_stripes_{tile_size_x}_{tile_size_y}")
    os.makedirs(src_output, exist_ok=True)

    xmin, xmax, ymin, ymax = np.min(laz.x), np.max(laz.x), np.min(laz.y), np.max(laz.y)

    # x_edges = np.arange(xmin, xmax, tile_size_x)
    # y_edges = np.arange(ymin, ymax, tile_size_y)


    # for i, x0 in enumerate(x_edges):
    #     print(f"Column {i+1} / {len(x_edges)}:")
    #     for j, y0 in tqdm(enumerate(y_edges), total=len(y_edges)):
    #         x1 = x0 + tile_size_x
    #         y1 = y0 + tile_size_y
    #         bounds = f"([{x0},{x1}],[{y0},{y1}])"
    #         output = os.path.join(src_output,f"{os.path.basename(src_input_file).split('.laz')[0]}_stripe_{i}_{j}.laz")
    pipeline_json = {
        "pipeline": [
            src_input_file,
            {
                "type": "filters.splitter",
                "length": stripe_width,
                "origin_x": xmin,
                "origin_y": ymin
            },
            {
                "type": "writers.las",
                "filename": os.path.join(src_output, "stripe_#.laz")
            }
        ]
    }


    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()


def stripes_one(src_input_file, src_output, dims, pos):
    # os.environ['PROJ_LIB'] = os.path.dirname(pyproj.__file__)
    i, j, x0, y0 = pos
    tile_size_x, tile_size_y = dims
    x1 = x0 + tile_size_x
    y1 = y0 + tile_size_y
    bounds = f"([{x0},{x1}],[{y0},{y1}])"
    output = os.path.join(src_output,f"{os.path.basename(src_input_file).split('.laz')[0]}_stripe_{i}_{j}.laz")
    pipeline_json = {
        "pipeline": [
            src_input_file,
            {"type": "filters.crop", "bounds": bounds},
            {"type": "writers.las", "filename": output, 'extra_dims': 'all'}
        ]
    }

    pipeline = pdal.Pipeline(json.dumps(pipeline_json))
    pipeline.execute()
    return 0


def stripes_multi(src_input_file, stripe_width):
    # src_folder_stripes = os.path.join(os.path.dirname(src_with_id), os.path.basename(src_with_id).split('.laz')[0] + "_stripes")
    src_output_folder = os.path.join(os.path.dirname(src_input_file), 'stripes')
    laz_original = laspy.read(src_input_file)
    x_span = laz_original.x.max() - laz_original.x.min()
    y_span = laz_original.y.max() - laz_original.y.min()
    dims = [x_span, y_span]

    dims[np.argmin([x_span, y_span])] = stripe_width

    os.makedirs(src_output_folder, exist_ok=True)

    xmin, xmax, ymin, ymax = np.min(laz_original.x), np.max(laz_original.x), np.min(laz_original.y), np.max(laz_original.y)

    x_span = laz_original.x.max() - laz_original.x.min()
    y_span = laz_original.y.max() - laz_original.y.min()
    dims = [x_span, y_span]

    dims[np.argmin([x_span, y_span])] = stripe_width
    tile_size_x, tile_size_y = dims

    x_edges = np.arange(xmin, xmax, tile_size_x)
    y_edges = np.arange(ymin, ymax, tile_size_y)

    list_pos = []
    time_start = time()
    for i, x0 in enumerate(x_edges):
        print(f"Column {i+1} / {len(x_edges)}:")
        for j, y0 in tqdm(enumerate(y_edges), total=len(y_edges)):
            list_pos.append((i,j,x0,y0))
            # stripes_one(src_input_file, src_output_folder, dims, (i, j, x0, y0))
    print("Duration without multi-processing: ", time() - time_start)

    # Multi-processing
    time_start = time()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        partial_func = partial(stripes_one, src_input_file, src_output_folder, dims)
        results = list(tqdm(executor.map(partial_func, list_pos), total=len(list_pos), smoothing=.9))
    print("Duration with multi-processing: ", time() - time_start)

# def stripe_one(src_input, dst_folder, stripe_width, xmin, xmax, ymin, ymax, j):
#     import pdal, json, os
#     y0 = ymin + j * stripe_width
#     y1 = y0 + stripe_width
#     bounds = f"([{xmin},{xmax}],[{y0},{y1}])"
#     output = os.path.join(dst_folder, f"stripe_{j:03d}.laz")

#     pipeline_json = {
#         "pipeline": [
#             src_input,
#             {"type": "filters.crop", "bounds": bounds},
#             {"type": "writers.las", "filename": output, "extra_dims": "all"}
#         ]
#     }
#     pdal.Pipeline(json.dumps(pipeline_json)).execute()
#     return output

def make_stripes_threaded(src_input, stripe_width):
    if "PROJ_LIB" not in os.environ:
        try:
            os.environ["PROJ_LIB"] = pyproj.datadir.get_data_dir()
        except Exception:
            # fallback paths
            for candidate in ["/usr/share/proj", "/opt/conda/share/proj"]:
                if os.path.exists(os.path.join(candidate, "proj.db")):
                    os.environ["PROJ_LIB"] = candidate
                    break
    dst_folder = os.path.join(os.path.dirname(src_input), os.path.basename(src_input).split('.laz')[0] + '_stripes')

    las = laspy.read(src_input)
    xmin, xmax, ymin, ymax = las.x.min(), las.x.max(), las.y.min(), las.y.max()
    os.makedirs(dst_folder, exist_ok=True)

    n_stripes = int(np.ceil((ymax - ymin) / stripe_width))
    with ThreadPoolExecutor(max_workers=8) as executor:
        func = partial(stripe_one, src_input, dst_folder, stripe_width, xmin, xmax, ymin, ymax)
        list(tqdm(executor.map(func, range(n_stripes)), total=n_stripes))


def main():
    src_input_file = r"data\test_stripes\Barmasse_2023_AllScans_Raw_Georef.laz"
    stripe_width = 20

    quit()
    stripes_multi(src_input_file, stripe_width)



if __name__ == "__main__":
    main()
