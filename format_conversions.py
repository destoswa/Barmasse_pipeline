import os
import sys
import numpy as np
from tqdm import tqdm
import json
import laspy
import traceback

class Convertions:
    @staticmethod
    def convert_laz_to_las(in_laz, out_las, offsets=[0,0,0], verbose=True):
        """
        Converts a LAZ file to an uncompressed LAS file.

        Args:
            - in_laz (str): Path to the input .laz file.
            - out_las (str): Path to the output .las file.
            - verbose (bool, optional): Whether to print confirmation of the saved file. Defaults to True.

        Returns:
            - None: Saves the converted .las file.
        """

        las = laspy.read(in_laz)
        las = laspy.convert(las)
        las.write(out_las)
        if verbose:
            print(f"LAS file saved in {out_las}")

    @staticmethod
    def convert_las_to_laz(in_las, out_laz, offsets=[0,0,0], verbose=True):
        """
        Converts a LAZ file to an uncompressed LAS file.

        Args:
            - in_laz (str): Path to the input .laz file.
            - out_las (str): Path to the output .las file.
            - verbose (bool, optional): Whether to print confirmation of the saved file. Defaults to True.

        Returns:
            - None: Saves the converted .las file.
        """

        las = laspy.read(in_las)
        las = laspy.convert(las)
        las.write(out_laz)
        if verbose:
            print(f"LAS file saved in {out_laz}")

    @staticmethod
    def convert_laz_to_txt(in_laz, out_txt, offsets=[0,0,0], verbose=True):
        laz = laspy.read(in_laz)
        list_dims = ['x', 'y', 'z','intensity']
        points = np.concatenate([np.array(getattr(laz, x)).reshape(-1,1) for x in list_dims], axis=1)
        points[:, 0] -= offsets[0]
        points[:, 1] -= offsets[1]
        points[:, 2] -= offsets[2]
        points[:, 3] =  points[:, 3] / 65535 * 255

        # Write to a txt file
        with open(out_txt, "w") as f:
            precision = int(np.log10(1/laz.header.scales[0]))
            np.savetxt(f, points, fmt=f"%.{precision}f,%.{precision}f,%.{precision}f,%d")

    @staticmethod
    def convert_txt_to_laz(in_txt, out_laz, offsets=[0,0,0], verbose=True):
        # Load the txt
        data = np.loadtxt(in_txt, delimiter=',')

        if data.ndim == 1:
            data = data.reshape(1, -1)
            
        with open(in_txt, 'r') as f:
            line  = f.readline()
        precision = len(line.split(',')[0].split('.')[-1])

        # Create a LAS file with a point format that supports intensity + extra dims
        header = laspy.LasHeader(point_format=3, version="1.4")
        header.scales = np.array([10**-precision, 10**-precision, 10**-precision])
        header.offsets = np.array(offsets)
        las = laspy.LasData(header)

        # Assign coordinates (laspy expects scaled ints internally)
        las.x = data[:, 0] + offsets[0]
        las.y = data[:, 1] + offsets[1]
        las.z = data[:, 2] + offsets[2]
        las.intensity = data[:, 3] / 255 * 65535

        # Save the result
        las.write(out_laz)

        if verbose:
            print(f"Saved LAZ with {len(las)} points to: {out_laz}")
            print("Dimensions:", las.point_format.dimension_names)


def convert_one_file(src_file_in, src_file_out, in_type, out_type, offsets=[0,0,0]):
    assert in_type in ['las', 'laz', 'txt']
    assert out_type in ['las', 'laz', 'txt']
    assert in_type != out_type

    if not hasattr(Convertions, f"convert_{in_type}_to_{out_type}"):
        print(f"No function for converting {in_type} into {out_type}!!")
        return
    try:
        _ = getattr(Convertions, f"convert_{in_type}_to_{out_type}")(src_file_in, src_file_out, offsets=offsets, verbose=False)
    except Exception as e:
        print(f"conversion from {in_type} to {out_type} for sample {src_file_in} failed")
        print(traceback.format_exc())
        pass


def convert_all_in_folder(src_folder_in, src_folder_out, in_type, out_type, offsets=[0,0,0], verbose=False):
    """
    Converts all files in a folder from one point cloud format to another.

    Args:
        - src_folder_in (str): Path to the input folder containing files to convert.
        - src_folder_out (str): Path to the output folder where converted files will be saved.
        - in_type (str): Input file type ('las', 'laz', or 'pcd').
        - out_type (str): Output file type ('las', 'laz', or 'pcd').
        - verbose (bool, optional): Whether to display a progress bar and detailed messages. Defaults to False.

    Returns:
        - None: Saves all converted files into the specified output folder.
    """
    
    assert in_type in ['las', 'laz', 'txt']
    assert out_type in ['las', 'laz', 'txt']
    assert in_type != out_type

    if not hasattr(Convertions, f"convert_{in_type}_to_{out_type}"):
        print(f"No function for converting {in_type} into {out_type}!!")
        return
    os.makedirs(src_folder_out, exist_ok=True)  # Ensure output folder exists
    files = [f for f in os.listdir(src_folder_in) if f.endswith(in_type)]
    for _, file in tqdm(enumerate(files), total=len(files), desc=f"Converting {in_type} in {out_type}", disable=verbose==False):
        file_out = file.split(in_type)[0] + out_type
        convert_one_file(os.path.join(src_folder_in, file), os.path.join(src_folder_out, file_out), in_type, out_type, offsets)


if __name__ == "__main__":
    if len(sys.argv) >= 5:
        src_folder_in = sys.argv[1]
        src_folder_out = sys.argv[2]
        in_type = sys.argv[3]
        out_type = sys.argv[4]
        verbose = False
        if len(sys.argv) == 6:
            if sys.argv[5].lower() == "true":
                verbose = True
        
        convert_all_in_folder(src_folder_in, src_folder_out, in_type, out_type, verbose)
    else:
        print("Missing arguments!")
        quit()