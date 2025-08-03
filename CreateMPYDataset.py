# CreateMPYDataset.py
import sys
import os
import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

# This allows the script to import your existing FireSpreadDataset class
sys.path.append(os.path.abspath(os.path.dirname(__file__).split("/src")[-2]))
from src.dataloader.FireSpreadDataset import FireSpreadDataset

# Need to prevent an error with HDF5 files being locked, which can also affect TIF reading
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

# --- Argument Parser (Same as your original script) ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str,
                    help="Path to the source dataset directory (containing TIFs)", required=True)
parser.add_argument("--target_dir", type=str,
                    help="Path to directory where the .mpy files should be stored", required=True)
args = parser.parse_args()

# --- Main Script Logic ---
print("Starting dataset conversion from TIF to MPY...")

# 1. Initialize the dataset to get access to the data generator
#    This reuses all the complex logic from FireSpreadDataset.py for finding and organizing files.
years = [2018, 2019, 2020, 2021]
dataset = FireSpreadDataset(
    data_dir=args.data_dir,
    included_fire_years=years,
    # The following arguments are irrelevant for this script but need to be set
    n_leading_observations=1,
    crop_side_length=128,
    load_from_hdf5=False,
    is_train=True,
    remove_duplicate_features=False,
    stats_years=(2018, 2019)
)
data_gen = dataset.get_generator_for_hdf5()

print("Found fire events. Starting conversion...")

# 2. Loop through each fire event provided by the generator
#    The generator yields a NumPy array 'imgs' for each fire, which is exactly what we need.
for year, fire_name, img_dates, lnglat, imgs in tqdm(data_gen, desc="Processing Fire Events"):

    # 3. Define the output path and create the directory if it doesn't exist
    target_year_dir = Path(args.target_dir) / str(year)
    target_year_dir.mkdir(parents=True, exist_ok=True)
    mpy_path = target_year_dir / f"{fire_name}.mpy"

    # 4. Check if the file already exists to avoid re-processing
    if mpy_path.is_file():
        # print(f"File {mpy_path} already exists, skipping...")
        continue

    # 5. Save the image data as an .mpy file using pickle
    #    The 'imgs' object is a NumPy array of shape (time, channels, height, width)
    #    which contains the full time series for one fire event.
    try:
        with open(mpy_path, "wb") as f:
            pickle.dump(imgs, f)
    except Exception as e:
        print(f"Could not write file for {fire_name}. Error: {e}")

print("Conversion to .mpy files complete!")
