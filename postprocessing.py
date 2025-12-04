import os
import traceback
import numpy as np
import laspy
from tqdm import tqdm
from time import time, sleep
from omegaconf import OmegaConf
from format_conversions import convert_all_in_folder, convert_one_file
from utils import add_matching_z, matching_mask, merge_laz
from playsound import playsound

def main(args):
    """
    Postprocessing of the pointcloud into final clean result.
    The process is subdivided into the following steps:
        - association of floor stripes and clean flatten stripes
        - extraction of clean stripes from original stripes
        - merging of clean stripes into big clean pointcloud
    """
    
    # hyperparameters
    SRC_INPUT = r'%s' %args.preprocessing.src_point_cloud
    INPUT_TYPE = SRC_INPUT.split('.')[-1].lower()
    SRC_CLEAN_STRIPES = args.postprocessing.src_clean_stripes
    SRC_FLOOR_STRIPES = args.postprocessing.src_floor_stripes
    SRC_ORIGINAL_STRIPES = args.postprocessing.src_original_stripes
    SRC_RESULTING_FILE = args.postprocessing.src_resulting_file
    OUTPUT_TYPE = args.postprocessing.output_type
    SKIP_TO_STEP = int(args.postprocessing.skip_to_step)

    with open(SRC_INPUT, 'rb') as f:
        scales = laspy.LasReader(f).header.scales
    ROUNDING  = int(-np.log10(scales[0]))

    # prepare paths
    src_postprocessing_files = os.path.join(os.path.dirname(SRC_CLEAN_STRIPES), "postprocessing_files")
    src_clean_laz_stripes = os.path.join(src_postprocessing_files, 'clean_stripes')
    src_results = os.path.join(src_postprocessing_files, 'results')
    src_merged_results = os.path.join(src_postprocessing_files, os.path.basename(SRC_INPUT).split(f'.{INPUT_TYPE}')[0] + "_merged.laz")

    list_stripes_to_merge = []

    os.makedirs(src_postprocessing_files, exist_ok=True)
    os.makedirs(src_results, exist_ok=True)

    # Conversion phase if clean tiles not in laz format
    if args.preprocessing.output_type != 'laz':
        print("\nConversion of clean stripes into laz for further processing")
        os.makedirs(src_clean_laz_stripes, exist_ok=True)
        if args.preprocessing.output_type == 'txt':
            if not os.path.exists(os.path.join(os.path.dirname(SRC_INPUT), 'offsets.txt')):
                raise FileExistsError("Missing offsets.txt file! Please rerun the preprocessing!")
            else:
                with open(os.path.join(os.path.dirname(SRC_INPUT), 'offsets.txt'), 'r') as f:
                    line = f.readline()
                offsets = [int(x) for x in line.split(',')]
        else:
            offsets = [0,0,0]

        convert_all_in_folder(SRC_CLEAN_STRIPES, src_clean_laz_stripes, args.preprocessing.output_type, 'laz', offsets, True)
        SRC_CLEAN_STRIPES = src_clean_laz_stripes
        print('\n' + '-'*20)

    # Step 1/3 - association of floor stripes and clean flatten stripes
    if SKIP_TO_STEP <= 1:
        print("\nStep 1/3 - Association of floor stripes and clean flatten stripes")
        clean_stripes = [x for x in os.listdir(SRC_CLEAN_STRIPES) if x.endswith('laz')]
        for _, stripe in tqdm(enumerate(clean_stripes), total=len(clean_stripes), desc="Processing"):
            src_result_clean_stripe = os.path.join(src_results, stripe.split('.laz')[0] + "_clean.laz")
            laz_flatten = laspy.read(os.path.join(SRC_CLEAN_STRIPES, stripe))
            laz_floor = laspy.read(os.path.join(SRC_FLOOR_STRIPES, stripe.replace('flatten', 'floor')))
            laz_original = laspy.read(os.path.join(SRC_ORIGINAL_STRIPES, stripe.replace('_flatten','')))

            assert int(-np.log10(laz_flatten.header.scales[0])) == ROUNDING

            xy_floor = np.column_stack([laz_floor.x, laz_floor.y])
            z_floor  = np.array(laz_floor.z)
            xyz_flatten = np.column_stack([laz_flatten.x, laz_flatten.y, laz_flatten.z])

            xyz_flatten = add_matching_z(xy_floor, z_floor, xyz_flatten, precision=ROUNDING)

            setattr(laz_flatten, 'z', xyz_flatten[:,2])
            laz_flatten.write(src_result_clean_stripe)

            list_stripes_to_merge.append(src_result_clean_stripe)

    # Step 2/3 - merging all flatten tiles together
    if SKIP_TO_STEP <= 2:
        print("\n---\n\nStep 2/3 - Merging all flatten tiles together")
        merge_laz(list_stripes_to_merge, src_merged_results)

    # Step 3/3 - filtering out the original file
    if SKIP_TO_STEP <= 3:
        print("\n---\n\nStep 3/3 - Filtering out the original file (might take a few minutes)")
        laz_original = laspy.read(SRC_INPUT)
        laz_results = laspy.read(src_merged_results)

        # extraction of clean stripes from original stripes
        xyz_original = np.column_stack([laz_original.x,laz_original.y,laz_original.z])
        xyz_clean = np.column_stack([laz_results.x, laz_results.y, laz_results.z])
        
        mask = matching_mask(xyz_original, xyz_clean, precision=ROUNDING)

        laz_original.points = laz_original.points[mask]

        if OUTPUT_TYPE in ['las', 'laz']:
            laz_original.write(SRC_RESULTING_FILE)
        else:
            laz_version_of_result = os.path.join(src_postprocessing_files, '.'.join(os.path.basename(SRC_RESULTING_FILE).split('.')[:-1]) + '.laz')
            laz_original.write(laz_version_of_result)
            convert_one_file(laz_version_of_result, SRC_RESULTING_FILE, 'laz', OUTPUT_TYPE)


if __name__ == "__main__":
    conf = OmegaConf.load('configs.yaml')
    
    # Save postprocessing conf
    if conf.general.do_save_conf:
        src_confs = os.path.join(os.path.dirname(conf.preprocessing.src_point_cloud), 'confs')
        os.makedirs(src_confs, exist_ok=True)
        with open(os.path.join(src_confs, 'postprocessing_conf.yaml'), 'w') as f:
            OmegaConf.save(conf.postprocessing, f.name)

    # Set paths if defaults
    src_preprocess_dir = os.path.join(os.path.dirname(conf.preprocessing.src_point_cloud), 'preprocessing_files')
    if conf.postprocessing.src_clean_stripes == "default":
        conf.postprocessing.src_clean_stripes = os.path.join(os.path.dirname(conf.preprocessing.src_point_cloud), 'to_process')
    if conf.postprocessing.src_floor_stripes == "default":
        conf.postprocessing.src_floor_stripes = os.path.join(src_preprocess_dir, os.path.basename(conf.preprocessing.src_point_cloud).split('.laz')[0] + f"_stripes_floor_{conf.preprocessing.stripe_width}_m")
    if conf.postprocessing.src_original_stripes == "default":
        conf.postprocessing.src_original_stripes = os.path.join(src_preprocess_dir, os.path.basename(conf.preprocessing.src_point_cloud).split('.laz')[0] + f"_stripes_original_{conf.preprocessing.stripe_width}_m")
    if conf.postprocessing.output_type == "default":
        conf.postprocessing.output_type = conf.preprocessing.src_point_cloud.split('.')[-1].lower()
    if conf.postprocessing.src_resulting_file == "default":
        conf.postprocessing.src_resulting_file = os.path.join(os.path.dirname(conf.preprocessing.src_point_cloud), os.path.basename(conf.preprocessing.src_point_cloud).split('.laz')[0] + f"_FINAL.{conf.postprocessing.output_type}")
    
    # Assertions
    assert conf.postprocessing.output_type in ['laz', 'las', 'txt']

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
    
    if conf.general.sound_when_finish:
        for i in range(3):
            sleep(0.5)
            playsound('./alarm.mp3')

    input("Press enter to continue...")