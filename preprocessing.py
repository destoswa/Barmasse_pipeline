import os
import shutil
import traceback
import numpy as np
import laspy
from tqdm import tqdm
from time import time
from omegaconf import OmegaConf
from format_conversions import convert_all_in_folder
# from utils import *
from utils_multi_proc import *
from playsound import playsound


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
    NUM_WORKERS = int(args.num_workers)

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
            n_processes=NUM_WORKERS,
            overlap=0)
    
    # 2 - splitting as tiles w overlap
    if SKIP_TO_STEP <= 2:
        print("\n---\n\nStep 2/11 - Tilling with overlap")
        tilling(
            src_input=SRC_INPUT, 
            src_target=src_folder_tiles_w_overlap, 
            tile_size=TILE_SIZE,
            n_processes=NUM_WORKERS,
            overlap=OVERLAP)
    
	# 3 - flattening of tiles w overlap
    if SKIP_TO_STEP <= 3:
        print("\n---\n\nStep 3/11 - Flattening of tiles")
        temp_time = time()
        flattening(
            src_tiles=src_folder_tiles_w_overlap,
            src_new_tiles=src_folder_flatten_tiles,
            grid_size=GRID_SIZE,
            method=METHOD,
            epsilon=METHOD_EPSILON,
            do_skip_existing=DO_SKIP_EXISTING_FLATTEN,
            n_processes=NUM_WORKERS,
            verbose=True,
            verbose_full=False,
        )
        print(f"Time to flatten = ", time() - temp_time)
	
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
        print("\n---\n\nStep 5/11 - Merging all flatten tiles together")
        merge_laz(list_flatten_to_merge, src_flatten_file)

	# 6 - creation of floor tiles wo overlap
    if SKIP_TO_STEP <= 6:
        print("\n---\n\nStep 6/11 - Creating floor tiles without overlap")
        list_floor_to_merge = []
        list_tiles = [x for x in os.listdir(src_folder_tiles_wo_overlap) if x.endswith('.laz')]
        for _, tile in tqdm(enumerate(list_tiles), total=len(list_tiles), desc="Processing"):
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
        print("\n---\n\nStep 7/11 - Merging all flatten tiles together")
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

    # Show configuration
    print(f"Preprocessing file: \n\t{conf.preprocessing.src_point_cloud} \nwith the following configuration:")
    for key, val in conf.preprocessing.items():
        if key != 'src_point_cloud':
            print(f"\t - {key}: {val}")
    print("-" * 20)

    start_preprocess_time = time()

    # Run main process
    try:
        main(conf.preprocessing)
    except Exception as e:
        print(traceback.format_exc())
        input("\nPress enter to continue...")
        quit()

    # Show duration of process
    delta_time_loop = time() - start_preprocess_time
    hours = int(delta_time_loop // 3600)
    min = int((delta_time_loop - 3600 * hours) // 60)
    sec = int(delta_time_loop - 3600 * hours - 60 * min)
    print(f"---\n\n==== Preprocessing done in {hours}:{min}:{sec} ====\n")

    # Show additional informations and instructions
    print("Additional information and instructions:")
    if conf.preprocessing.output_type == 'txt':
        # Load header of input to extract precision used
        with open(conf.preprocessing.src_point_cloud, 'rb') as file:
            scales = laspy.LasReader(file).header.scales
        precision = int(np.log10(1/scales[0]))
        
        # Show information/instructions
        print("\tThe output have been centered in order to avoid errors linked to rounding errors. " \
        "The offsets were saved in the file data/offsets.txt and will automatically be loaded by the postprocess.")
        print(f"\tThe precision detected and used is {precision} decimals. It is important that the " \
        'clean files given to postprocess have the same precision. ')
    
    for i in range(3):
        playsound('./robot.mp3')
    
    input("Press enter to continue...")
