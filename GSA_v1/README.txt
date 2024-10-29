### `README.txt`

```
# GSA_ver.1

## Authors: A. Velichko, M. Belyaev, I. A. Oludehinwa and O.I. Olusola

This program processes solar image data, applies feature extraction, and visualizes the results with color masks and boundaries.

## Instructions

### 1. Install Required Libraries

Run the `install.bat` file to install the necessary Python packages. Here is the content of `install.bat`:

```batch
@echo off
echo Installing required Python packages...

pip install numpy matplotlib sunpy scikit-image joblib EntropyHub

if %errorlevel% neq 0 (
    echo.
    echo There was an error during the installation of the packages.
    exit /b %errorlevel%
)

echo.
echo All packages installed successfully.
PAUSE
```

### 2. Prepare FITS Files

Place the FITS files containing solar images in the `solar_fits_files` folder. Each FITS file should be in its own subfolder. The main solar image file should be named `image.fits`.

Optionally, you can fill the `boundaries.txt` file with boundary data, for example, downloaded from the HEK website. If not needed, leave this file empty.

### 3. Prepare Kernel Files

The `kernels` folder contains data for circular kernels of various radii (R) from 1 to 28 pixels. The default radius is R=14.

### 4. Configure Processing Parameters

Before running the image processing, configure the processing parameters in the `parameters.txt` file. The file should contain the following fields:

```text
folder_name = solar_fits_files\2021_06_05
file_name = image.fits
boundaries_file_name = boundaries.txt
kernel_folder_name = kernels
kernel_file_name = 14_kernel.txt
model_filename = svc_model.joblib
model_filename2 = direct_clf2.joblib
initial_brightness = 80
num_threads = 2
chunksize = 30
use_second_model = True
```

- `folder_name`: Name of the folder containing the FITS file.
- `file_name`: Name of the FITS file.
- `boundaries_file_name`: Name of the file with boundary data.
- `kernel_file_name`: Name of the file with circular kernel data.
- `model_filename`: Name of the first SVC model file working with entropy features.
- `model_filename2`: Name of the second SVC model file working directly with the time series obtained from the circular kernel.
- `initial_brightness`: Initial brightness for image rendering (can be adjusted after displaying the image).
- `num_threads`: Number of threads to use for model calculation.
- `chunksize`: Number of samples to load per chunk when working with the first model.
- `use_second_model`: Set to `True` to use the second model, `False` to use the first model.

### 5. Run the Main Calculation

Run the `gsa.py` script to start the main calculation. It is recommended to run `gsa.bat`.

After the script completes, the result files will be output to the `folder_name` directory. Besides images, the results include the `output_log.txt` file, which contains key information about the model used and the calculation results for GSA values. The log also includes various calculation times, such as the total calculation time with the specified number of threads and the per-thread calculation time.

### Additional Information

For calculating multiple images, run the `run_all.py` file, which can be executed using `run_all.bat`. Before doing so, specify the paths and parameters for all calculations in the `run_all.txt` file. This file has a structure similar to the `parameters.txt` file.

### Citation

When using this program, please cite the original article:
"Title of the Article"

Solar Active Regions Detection Via 2D Circular Kernel Time Series Transformation, Entropy and Machine Learning Approach

This `README.txt` provides clear instructions for users on how to install the necessary libraries, prepare the required files, configure the processing parameters, run the main calculation, and calculate multiple images. It also includes a section for citing the original article.