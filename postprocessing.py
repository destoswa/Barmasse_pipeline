import os
import traceback
import numpy as np
import laspy
from tqdm import tqdm
from time import time
from omegaconf import OmegaConf
from format_conversions import convert_all_in_folder, convert_one_file


# Utils

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


def main(args):
    """
    Postprocessing of the pointcloud into final clean result.
    The process is subdivided into the following steps:
        - association of floor stripes and clean flatten stripes
        - extraction of clean stripes from original stripes
        - merging of clean stripes into big clean pointcloud
    """
    
    # hyperparameters
    SRC_CLEAN_STRIPES = args.postprocessing.src_clean_stripes
    SRC_FLOOR_STRIPES = args.postprocessing.src_floor_stripes
    SRC_ORIGINAL_STRIPES = args.postprocessing.src_original_stripes
    SRC_RESULTING_FILE = args.postprocessing.src_resulting_file
    SKIP_TO_STEP = int(args.postprocessing.skip_to_step)
    ROUNDING_FLOOR_FLATTEN  = int(args.postprocessing.rounding_floor_flatten)
    ROUNDING_CLEAN_FINAL  = int(args.postprocessing.rounding_clean_final)

    # prepare paths
    src_postprocessing_files = os.path.join(os.path.dirname(SRC_CLEAN_STRIPES), "postprocessing_files")
    src_clean_laz_stripes = os.path.join(src_postprocessing_files, 'clean_stripes')
    src_results = os.path.join(src_postprocessing_files, 'results')

    list_stripes_to_merge = []

    os.makedirs(src_postprocessing_files, exist_ok=True)
    os.makedirs(src_results, exist_ok=True)

    # Conversion phase if clean tiles not in laz format
    if args.preprocessing.output_type != 'laz':
        print("Conversion of clean stripes into laz for further processing")
        os.makedirs(src_clean_laz_stripes, exist_ok=True)
        if args.preprocessing.output_type == 'txt':
            if not os.path.exists(os.path.join(os.path.dirname(args.preprocessing.src_point_cloud), 'offsets.txt')):
                raise FileExistsError("Missing offsets.txt file! Please rerun the preprocessing!")
            else:
                with open(os.path.join(os.path.dirname(args.preprocessing.src_point_cloud), 'offsets.txt'), 'r') as f:
                    line = f.readline()
                offsets = [int(x) for x in line.split(',')]
        else:
            offsets = [0,0,0]

        convert_all_in_folder(SRC_CLEAN_STRIPES, src_clean_laz_stripes, args.preprocessing.output_type, 'laz', offsets, True)
        SRC_CLEAN_STRIPES = src_clean_laz_stripes

    # Step 1/2 - association of floor stripes and clean flatten stripes
    if SKIP_TO_STEP <= 1:
        print("\nStep 1/2 - Association of floor stripes and clean flatten stripes")
        clean_stripes = [x for x in os.listdir(SRC_CLEAN_STRIPES) if x.endswith('laz')]
        for _, stripe in tqdm(enumerate(clean_stripes), total=len(clean_stripes), desc="Processing"):
            src_result_clean_tile = os.path.join(src_results, stripe.split('.laz')[0] + "_clean.laz")
            src_result_final_tile = os.path.join(src_results, stripe.split('.laz')[0] + "_FINAL.laz")
            laz_flatten = laspy.read(os.path.join(SRC_CLEAN_STRIPES, stripe))
            laz_floor = laspy.read(os.path.join(SRC_FLOOR_STRIPES, stripe.replace('flatten', 'floor')))
            laz_original = laspy.read(os.path.join(SRC_ORIGINAL_STRIPES, stripe.replace('_flatten','')))

            xy_floor = np.column_stack([laz_floor.x, laz_floor.y])
            z_floor  = np.array(laz_floor.z)

            xyz_flatten = np.column_stack([laz_flatten.x, laz_flatten.y, laz_flatten.z])

            xyz_flatten = add_matching_z(xy_floor, z_floor, xyz_flatten, precision=ROUNDING_FLOOR_FLATTEN)

            setattr(laz_flatten, 'z', xyz_flatten[:,2])

            laz_flatten.write(src_result_clean_tile)

            # extraction of clean stripes from original stripes
            xyz_original = np.column_stack([laz_original.x,laz_original.y,laz_original.z])
            xyz_clean = np.column_stack([laz_flatten.x, laz_flatten.y, laz_flatten.z])

            mask = matching_mask(xyz_original, xyz_clean, precision=ROUNDING_CLEAN_FINAL)

            laz_original.points = laz_original.points[mask]
            laz_original.write(src_result_final_tile)
            matching_frac = 100 - len(laz_original)/len(laz_flatten)*100
            if matching_frac > 1.0:
                print(f"Alert with file {stripe}!\n\tMore than 1% points of the clean files are not matching the original ({round(matching_frac, 2)}%)")
            
            list_stripes_to_merge.append(src_result_final_tile)

    # Step 2/2 - merging all flatten tiles together (might take a few minutes)
    if SKIP_TO_STEP <= 2:
        print("\n---\n\nStep 2/2 - Merging all flatten tsiles together (might take a few minutes)")
        merge_laz(list_stripes_to_merge, SRC_RESULTING_FILE)


if __name__ == "__main__":
    conf = OmegaConf.load('configs.yaml')
    
    # Set paths if defaults
    src_preprocess_dir = os.path.join(os.path.dirname(conf.preprocessing.src_point_cloud), 'preprocessing_files')
    if conf.postprocessing.src_clean_stripes == "default":
        conf.postprocessing.src_clean_stripes = os.path.join(os.path.dirname(conf.preprocessing.src_point_cloud), 'to_process')
    if conf.postprocessing.src_floor_stripes == "default":
        conf.postprocessing.src_floor_stripes = os.path.join(src_preprocess_dir, os.path.basename(conf.preprocessing.src_point_cloud).split('.laz')[0] + f"_stripes_floor_{conf.preprocessing.stripe_width}_m")
    if conf.postprocessing.src_original_stripes == "default":
        conf.postprocessing.src_original_stripes = os.path.join(src_preprocess_dir, os.path.basename(conf.preprocessing.src_point_cloud).split('.laz')[0] + f"_stripes_original_{conf.preprocessing.stripe_width}_m")
    if conf.postprocessing.src_resulting_file == "default":
        conf.postprocessing.src_resulting_file = os.path.join(os.path.dirname(conf.preprocessing.src_point_cloud), os.path.basename(conf.preprocessing.src_point_cloud).split('.laz')[0] + "_FINAL.laz")

    print(f"Preprocessing files from : - \n\t{conf.postprocessing.src_clean_stripes}\n\t{conf.postprocessing.src_floor_stripes} \n\t{conf.postprocessing.src_original_stripes} \nwith the following configuration:")
    for key, val in conf.postprocessing.items():
        if key not in ["src_clean_stripes", "src_floor_stripes", "src_original_stripes"]:
            print(f"\t - {key}: {val}")
    print("-" * 20)

    start_postprocess_time = time()

    try:
        main(conf)
    except Exception as e:
        print(traceback.format_exc())
        input("\nPress enter to continue...")
        quit()

    # Showing duration of process
    delta_time_loop = time() - start_postprocess_time
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"\n==== Postprocessing done in {hours}:{min}:{sec} ====\n")

    input("Press enter to continue...")