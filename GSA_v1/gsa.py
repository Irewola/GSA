"""
GSA_ver.1
Authors: A. Velichko, M. Belyaev, and I. A. Oludehinwa

This program processes solar image data, applies feature extraction, 
and visualizes the results with color masks and boundaries.
"""

import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
import sunpy.map
import ast
from skimage.draw import line as draw_line
from joblib import load
import EntropyHub as EH
import multiprocessing as mp
import argparse

# Configurable folder and file names
# Function to read parameters from a file
def read_parameters(file_path):
    parameters = {}
    with open(file_path, 'r') as file:
        for line in file:
            if '=' in line:
                key, value = line.strip().split('=', 1)
                parameters[key.strip()] = value.strip()
    return parameters

# Read parameters from the external file
params = read_parameters('parameters.txt')

# Configurable folder and file names
folder_name = params.get('folder_name', 'default_folder')
file_name = params.get('file_name', 'default_file.fits')
boundaries_file_name = params.get('boundaries_file_name', 'boundaries.txt')
kernel_folder_name = params.get('kernel_folder_name', 'kernels')
kernel_file_name = params.get('kernel_file_name', '14_kernel.txt')
model_filename = params.get('model_filename', 'svc_model.joblib')
model_filename2 = params.get('model_filename2', 'direct_clf2.joblib')
initial_brightness = int(params.get('initial_brightness', 80))
num_threads = int(params.get('num_threads', 1))
chunksize = int(params.get('chunksize', 1))
use_second_model = params.get('use_second_model', 'False').lower() == 'true'

# Set up argument parser
parser = argparse.ArgumentParser(description="Run the script with or without plotting.")
parser.add_argument('--auto', action='store_true', help="Run without displaying the plot.")
args = parser.parse_args()

# Path to the calculation completed file
calculation_completed_file = 'calculation_completed.txt'

# Remove the calculation completed file if it exists at the start
if os.path.exists(calculation_completed_file):
    os.remove(calculation_completed_file)

# Backup original stdout
original_stdout = sys.stdout
unix_time = int(time.time())
# Open a file for logging
log_file_path = os.path.join(folder_name, str(unix_time)+'_output_log.txt')

def log_print(message):
    """
    Prints a message to the console and writes it to a log file.
    """
    print(message)
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(message + '\n')
        log_file.flush()

def read_matrix_from_file(folder, file):
    """
    Reads the matrix data from a FITS file and extracts solar parameters.
    """
    file_path = os.path.join(folder, file)
    aia_map = sunpy.map.Map(file_path)
    solar_radius = aia_map.fits_header['R_SUN']
    center_x = aia_map.fits_header['CRPIX1']
    center_y = aia_map.fits_header['CRPIX2']
    log_print(f'solar_radius={solar_radius}')
    log_print(f'center_x={center_x}')
    log_print(f'center_y={center_y}')
    data = aia_map.data
    return np.array(data), solar_radius, center_x, center_y

def normalize_matrix(matrix, min_val, max_val):
    """
    Normalizes the matrix values between 0 and 1 based on given min and max values.
    """
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    normalized_matrix[normalized_matrix < 0] = 0
    normalized_matrix[normalized_matrix > 1] = 1
    return normalized_matrix

def read_kernel(folder, file):
    """
    Reads kernel data from a file.
    """
    file_path = os.path.join(folder, file)
    with open(file_path, 'r') as f:
        lines = f.readlines()
    int_count = int(lines[0].strip())
    x_int_ar = []
    y_int_ar = []
    for line in lines[1:int_count + 1]:
        x, y = map(int, line.split('\t'))
        x_int_ar.append(x)
        y_int_ar.append(y)
    return int_count, x_int_ar, y_int_ar

def get_fuzz_entropy(x, p1, p2, p3):
    """
    Calculates the Fuzzy Entropy of a time series.
    """
    ent, _, _ = EH.FuzzEn(x, m=p1, r=(p2 * np.std(x, ddof=0), p3), Fx='default')
    if not np.isfinite(ent[-1]):
        return 0
    return ent[-1]

def get_dist_ent(x, p1, p2):
    """
    Calculates the Distribution Entropy of a time series.
    """
    ent, _ = EH.DistEn(x, m=p1, Bins=p2)
    if not np.isfinite(ent) or np.isnan(ent):
        return 0
    return ent

def block_print():
    """
    Disables print output.
    """
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    """
    Restores print output.
    """
    sys.stdout = original_stdout

def generate_features(series):
    """
    Generates features from a time series.
    """
    result = []
    block_print()
    result.append(get_dist_ent(series, 2, 'sqrt'))
    result.append(get_fuzz_entropy(series, 1, 0.2, 5))
    result.append(np.median(series))
    result.append(np.nanpercentile(series, 95))
    enable_print()
    return np.array(result).reshape(1, -1)

def tser(matrix, x, y, int_count, x_int_ar, y_int_ar):
    """
    Generates a time series from a matrix given coordinates and kernel data.
    """
    arr_max_2d = matrix.shape[0]
    ser = np.zeros(int_count, dtype=int)
    y0 = y
    x0 = x
    for i2 in range(int_count):
        y_int = y0 + y_int_ar[i2]
        x_int = x0 + x_int_ar[i2]
        if x_int >= arr_max_2d:
            x_int = arr_max_2d - (x_int - arr_max_2d) - 1
        if y_int >= arr_max_2d:
            y_int = arr_max_2d - (y_int - arr_max_2d) - 1
        if x_int < 0:
            x_int = -x_int
        if y_int < 0:
            y_int = -y_int
        ser[i2] = matrix[y_int, x_int]
    return ser

def calculate_total_points(arr_max_2d, center_x, center_y, solar_radius):
    """
    Calculates the total number of valid points within the solar radius.
    """
    total_points = 0
    for j in range(7, arr_max_2d, 8):
        for i in range(7, arr_max_2d, 8):
            if np.sqrt((i - center_x) ** 2 + (j - center_y) ** 2) <= solar_radius:
                total_points += 1
    return total_points

def process_point(args):
    """
    Processes a single point to generate features.
    """
    i, j, matrix, int_count, x_int_ar, y_int_ar = args
    ser = tser(matrix, i, j, int_count, x_int_ar, y_int_ar)
    X = generate_features(ser)
    return (i, j, X)

def progress_updater(total_tasks, current_point):
    """
    Updates progress of the processing tasks.
    """
    old_value = current_point.value
    while current_point.value < total_tasks:
        time.sleep(1)
        new_value = current_point.value
        progress = (new_value / total_tasks) * 100
        print(f"Progress: {progress:.2f}% ({new_value}/{total_tasks})")
        print(f"Increment: {new_value - old_value}")
        old_value = new_value
    print(f"Progress: 100.00% ({total_tasks}/{total_tasks})")

def generate_time_series(matrix, int_count, x_int_ar, y_int_ar, mask, svc_model, solar_radius, center_x, center_y, output_file='ser.txt', num_threads=30):
    """
    Generates time series data and processes it to classify each point.
    """
    arr_max_2d = matrix.shape[0]
    total_points = calculate_total_points(arr_max_2d, center_x, center_y, solar_radius)
    log_print(f"total_points = {total_points}")

    current_point = mp.Value('i', 0)
    start_time = time.time()

    tasks = [(i, j, matrix, int_count, x_int_ar, y_int_ar) for j in range(7, arr_max_2d, 8) for i in range(7, arr_max_2d, 8) if np.sqrt((i - center_x)**2 + (j - center_y)**2) <= solar_radius]

    # Initialize feature_matrix
    feature_matrix = np.zeros((arr_max_2d, arr_max_2d), dtype=object)

    # Start progress updater thread
    progress_thread = mp.Process(target=progress_updater, args=(len(tasks), current_point))
    progress_thread.start()

    # Create a pool of processes and submit tasks
    with mp.Pool(num_threads) as pool:
        for result in pool.imap_unordered(process_point, tasks, chunksize=chunksize):
            i, j, X = result
            feature_matrix[j, i] = X
            with current_point.get_lock():
                current_point.value += 1

    # Wait for progress updater to finish
    progress_thread.join()
    log_print(f"Features calculated")

    # Initialize counters for each class
    class_nAR_type_1 = 0
    class_nAR_type_2 = 0
    class_AR = 0

    current_point = 0
    start_time = time.time()

    # Second loop to predict classes and update mask
    for j in range(7, arr_max_2d, 8):
        for i in range(7, arr_max_2d, 8):
            if feature_matrix[j, i] is not None and isinstance(feature_matrix[j, i], np.ndarray):
                X = feature_matrix[j, i]
                y_pred = svc_model.predict(X)[0]
                mask[j, i] = y_pred

                # Increment the appropriate class counter
                if y_pred == 0:
                    class_nAR_type_1 += 1
                elif y_pred == 1:
                    class_nAR_type_2 += 1
                elif y_pred == 2:
                    class_AR += 1

                current_point += 1
                if time.time() - start_time >= 1:
                    progress = (current_point / total_points) * 100
                    print(f"Progress: {progress:.2f}%")
                    print(f'class_nAR_type_1={class_nAR_type_1}')
                    print(f'class_nAR_type_2={class_nAR_type_2}')
                    print(f'class_AR={class_AR}')
                    start_time = time.time()

    log_print(f"Class_nAR_type_1 count: {class_nAR_type_1}")
    log_print(f"Class_nAR_type_2: {class_nAR_type_2}")
    log_print(f"Class_AR count: {class_AR}")
    log_print('Generalized Solar Activity for class_AR: (class_AR) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)')
    log_print(f"Generalized Solar Activity for class_AR: {(class_AR) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)}")
    log_print('Generalized Solar Activity for class_nAR_type_2: (class_nAR_type_2) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)')
    log_print(f"Generalized Solar Activity for class_nAR_type_2: {(class_nAR_type_2) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)}")
    log_print('Generalized Solar Activity for class_nAR_type_2+class_AR: (class_nAR_type_2+class_AR) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)')
    log_print(f"Generalized Solar Activity for class_nAR_type_2+class_AR: {(class_nAR_type_2+class_AR) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)}")

def worker(task_chunk, int_count, x_int_ar, y_int_ar, matrix, svc_model):
    
    ser_arr = []
    task_indices = []
    for i, j in task_chunk:
        ser = tser(matrix, i, j, int_count, x_int_ar, y_int_ar)
        ser_arr.append(ser)
        task_indices.append((i, j))
        
    
    ser_arr = np.array(ser_arr)
    
    
    
    y_pred_arr = svc_model.predict(ser_arr)
    
    
    return list(zip(task_indices, y_pred_arr))

def chunk_tasks(tasks, num_chunks):
    for i in range(0, len(tasks), num_chunks):
        yield tasks[i:i + num_chunks]

def generate_time_series2(matrix, int_count, x_int_ar, y_int_ar, mask, svc_model, solar_radius, center_x, center_y, num_threads=30):
    """
    Generates time series data and processes it to classify each point using the second model.
    """
    start_time_total = time.time()
    
    arr_max_2d = matrix.shape[0]
    start_time = time.time()
    total_points = calculate_total_points(arr_max_2d, center_x, center_y, solar_radius)
    log_print(f"total_points = {total_points}")
    log_print(f"Time to calculate total points: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    tasks = [(i, j) 
             for j in range(7, arr_max_2d, 8) 
             for i in range(7, arr_max_2d, 8) 
             if np.sqrt((i - center_x)**2 + (j - center_y)**2) <= solar_radius]
    log_print(f"Time to generate tasks: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    # Process chunks in parallel
    with mp.Pool(num_threads) as pool:
        results = pool.starmap(worker, [(chunk, int_count, x_int_ar, y_int_ar, matrix, svc_model) for chunk in chunk_tasks(tasks, len(tasks) // num_threads + 1)])
    log_print(f"Time to process tasks in parallel: {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    # Update mask with predictions
    for result in results:
        for (i, j), y_pred in result:
            mask[j, i] = y_pred
    log_print(f"Time to update mask with predictions: {time.time() - start_time:.2f} seconds")

    # Initialize counters for each class
    class_nAR_type_1 = 0
    class_nAR_type_2 = 0
    class_AR = 0

    start_time = time.time()
    # Count the number of each class in the mask
    for i, j in tasks:
        y_pred = mask[j, i]
        if y_pred == 0:
            class_nAR_type_1 += 1
        elif y_pred == 1:
            class_nAR_type_2 += 1
        elif y_pred == 2:
            class_AR += 1
    log_print(f"Time to count classes in the mask: {time.time() - start_time:.2f} seconds")

    log_print(f"Class_nAR_type_1 count: {class_nAR_type_1}")
    log_print(f"Class_nAR_type_2: {class_nAR_type_2}")
    log_print(f"Class_AR count: {class_AR}")
    log_print('Generalized Solar Activity for class_AR: (class_AR) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)')
    log_print(f"Generalized Solar Activity for class_AR: {(class_AR) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)}")
    log_print('Generalized Solar Activity for class_nAR_type_2: (class_nAR_type_2) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)')
    log_print(f"Generalized Solar Activity for class_nAR_type_2: {(class_nAR_type_2) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)}")
    log_print('Generalized Solar Activity for class_nAR_type_2+class_AR: (class_nAR_type_2+class_AR) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)')
    log_print(f"Generalized Solar Activity for class_nAR_type_2+class_AR: {(class_nAR_type_2+class_AR) * 100 / (class_nAR_type_1+class_nAR_type_2+class_AR)}")


    log_print(f"Total execution time: {time.time() - start_time_total:.2f} seconds")






def expand_mask(mask, size=8):
    """
    Expands the mask by a given size.
    """
    arr_max_2d = mask.shape[0]
    expanded_mask = mask.copy()
    offset = size // 2
    for j in range(arr_max_2d):
        for i in range(arr_max_2d):
            if mask[j, i] in [1, 2]:
                for y in range(max(0, j - offset), min(arr_max_2d, j + offset + 1)):
                    for x in range(max(0, i - offset), min(arr_max_2d, i + offset + 1)):
                        expanded_mask[y, x] = mask[j, i]
    for j in range(arr_max_2d):
        for i in range(arr_max_2d):
            if mask[j, i] in [3]:
                for y in range(max(0, j - offset), min(arr_max_2d, j + offset + 1)):
                    for x in range(max(0, i - offset), min(arr_max_2d, i + offset + 1)):
                        expanded_mask[y, x] = mask[j, i]
    return expanded_mask

def apply_mask(matrix, mask, apply_color_mask):
    """
    Applies the mask to the matrix, optionally using colors.
    """
    color_image = plt.cm.gray(matrix)
    if apply_color_mask:
        color_image[mask == 1] = [0, 0, 1, 1]  # Blue for class 1
        color_image[mask == 2] = [1, 1, 0, 1]  # Yellow for class 2
        color_image[mask == 3] = [1, 0, 0, 1]  # Red for class 3
    return color_image
    
def apply_mask2(matrix, mask, apply_color_mask):
    """
    Applies the mask to the matrix, optionally using colors.
    """
    color_image = plt.cm.gray(matrix)
    if apply_color_mask:
        #color_image[mask == 1] = [0, 0, 1, 1]  # Blue for class 1
        #color_image[mask == 2] = [1, 1, 0, 1]  # Yellow for class 2
        color_image[mask == 3] = [1, 0, 0, 1]  # Red for class 3
    return color_image
    
def save_image(color_image, filename):
    """
    Saves the image with the applied mask to a file.
    """
    plt.imsave(filename, color_image)

def draw_boundaries(mask, boundaries_file):
    """
    Draws boundaries on the mask from a boundaries file.
    """
    with open(boundaries_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        points = ast.literal_eval(line.strip())
        points = np.array(points)
        for i in range(len(points)):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % len(points)]
            rr, cc = draw_line(int(x1), int(y1), int(x2), int(y2))
            mask[rr, cc] = 3
    return mask

def update(val):
    """
    Updates the visualization based on the slider value.
    """
    brightness = slider.val
    min_val = min_value
    max_val = max_value - (max_value - min_value) * (brightness / 100.0)
    normalized_matrix = normalize_matrix(matrix, min_val, max_val)
    masked_image = apply_mask(normalized_matrix, mask, check.get_status()[0])
    img.set_data(masked_image)
    fig.canvas.draw_idle()
    
def save_mask_to_file(mask, folder_name, file_name):
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, 'w') as f:
        for row in mask:
            f.write(' '.join(map(str, row)) + '\n')
            
def main():
    log_print('///////////////////START///////////////////')
    global matrix, min_value, max_value, slider, img, fig, mask, check

    # Load the appropriate model
    if use_second_model:
        svc_model = load(model_filename2)
        log_print(f"USE SECOND MODEL")
        log_print(f"Model filename: {model_filename2}")
        generate_time_series_func = generate_time_series2
    else:
        svc_model = load(model_filename)
        log_print(f"USE FIRST MODEL")
        log_print(f"Model filename: {model_filename}")
        generate_time_series_func = generate_time_series

    log_print('unix_time='+str(unix_time))
    log_print(f"Folder name: {folder_name}")
    log_print(f"File name: {file_name}")
    log_print(f"Boundaries file name: {boundaries_file_name}")
    log_print(f"Kernel folder name: {kernel_folder_name}")
    log_print(f"Kernel file name: {kernel_file_name}")
    log_print(f"Initial brightness: {initial_brightness}")
    log_print(f"Number of threads: {num_threads}")
    log_print(f"Chunksize: {chunksize}")

    # Dictionary to store time taken for each step
    time_dict = {}

    # Ensure the folder exists
    start_time = time.time()
    if not os.path.exists(folder_name):
        log_print(f"Folder '{folder_name}' does not exist.")
        return
    time_dict['Check folder exists'] = time.time() - start_time

    # Ensure the file exists
    start_time = time.time()
    file_path = os.path.join(folder_name, file_name)
    if not os.path.exists(file_path):
        log_print(f"File '{file_name}' does not exist in the folder '{folder_name}'.")
        return
    time_dict['Check file exists'] = time.time() - start_time

    # Ensure the kernel folder and file exist
    start_time = time.time()
    kernel_file_path = os.path.join(kernel_folder_name, kernel_file_name)
    if not os.path.exists(kernel_file_path):
        log_print(f"Kernel file '{kernel_file_name}' does not exist in the folder '{kernel_folder_name}'.")
        return
    time_dict['Check kernel file exists'] = time.time() - start_time

    # Read the matrix from the file and get solar parameters
    start_time = time.time()
    matrix, solar_radius, center_x, center_y = read_matrix_from_file(folder_name, file_name)
    time_dict['Read matrix from file'] = time.time() - start_time

    # Print the dimensions of the matrix
    log_print(f"Matrix dimensions: {matrix.shape}")

    # Find and print the minimum and maximum values in the matrix
    start_time = time.time()
    min_value = np.min(matrix)
    max_value = np.max(matrix)
    time_dict['Find min and max values'] = time.time() - start_time
    log_print(f"Minimum value in the matrix: {min_value}")
    log_print(f"Maximum value in the matrix: {max_value}")

    # Calculate the initial normalized matrix based on the initial brightness
    start_time = time.time()
    initial_max_val = max_value - (max_value - min_value) * (initial_brightness / 100.0)
    normalized_matrix = normalize_matrix(matrix, min_value, initial_max_val)
    time_dict['Normalize matrix'] = time.time() - start_time

    # Initialize mask with zeros (class 0)
    start_time = time.time()
    mask = np.zeros(matrix.shape, dtype=int)
    time_dict['Initialize mask'] = time.time() - start_time

    # Read the kernel data
    start_time = time.time()
    int_count, x_int_ar, y_int_ar = read_kernel(kernel_folder_name, kernel_file_name)
    time_dict['Read kernel data'] = time.time() - start_time

    # Measure and print the time taken by generate_time_series
    start_time = start_time_total = time.time()
    generate_time_series_func(matrix, int_count, x_int_ar, y_int_ar, mask, svc_model, solar_radius, center_x, center_y, num_threads=num_threads)
    time_dict['Generate time series'] = time.time() - start_time
    time_dict['Generate time series per 1 thread'] =  time_dict['Generate time series']*num_threads

    # Draw boundaries from file
    boundaries_file_path = os.path.join(folder_name, boundaries_file_name)
    if os.path.exists(boundaries_file_path):
        start_time = time.time()
        mask = draw_boundaries(mask, boundaries_file_path)
        time_dict['Draw boundaries'] = time.time() - start_time
        log_print('boundaries_file_path='+str(boundaries_file_path))
    else:
        log_print(f"File not found: {boundaries_file_path}")    
    
    # Expand the mask to 5x5 squares
    start_time = time.time()
    mask = expand_mask(mask)
    time_dict['Expand mask'] = time.time() - start_time
    # Save the initial image
    
    masked_image = apply_mask(normalized_matrix, None, True)
    save_image(masked_image, os.path.join(folder_name, str(unix_time)+'_initial_image.jpg'))

    # Apply the mask with the desired color (red in this case)
    start_time = time.time()
    masked_image = apply_mask2(normalized_matrix, mask, True)
    time_dict['Apply  boundaries mask to image'] = time.time() - start_time

    # Save the masked image
    save_image(masked_image, os.path.join(folder_name, str(unix_time)+'_initial_image_boundaries.jpg'))
    
    # Apply the mask with the desired color (red in this case)
    start_time = time.time()
    masked_image = apply_mask(normalized_matrix, mask, True)
    time_dict['Apply mask to image'] = time.time() - start_time

    # Save the masked image
    model_suffix = '_second_model' if use_second_model else '_first_model'
    save_image(masked_image, os.path.join(folder_name, f"{str(unix_time)}_initial_image_boundaries{model_suffix}.jpg"))
    
    # Save the mask to a file with elements separated by spaces
    #mask_file_name = str(unix_time) + '_mask.txt'
    #save_mask_to_file(mask, folder_name, mask_file_name)  
    
    # Print the execution time for each step
    total_execution_time = time.time() - start_time_total
    log_print("Execution time for each step:")
    for step, duration in time_dict.items():
        log_print(f"{step}: {duration:.2f} seconds")

    log_print(f"Total execution time: {total_execution_time:.2f} seconds")

    # Create the plot
    start_time = time.time()
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)
    img = ax.imshow(masked_image, interpolation='nearest', cmap='gray')
    plt.colorbar(img, ax=ax)
    plt.title('Color Representation of the Matrix with Mask')
    ax_slider = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Brightness', 10, 100, valinit=initial_brightness, valstep=1)
    slider.on_changed(update)

    # Add CheckButtons for toggling color mask
    ax_check = plt.axes([0.025, 0.5, 0.2, 0.15], facecolor='lightgoldenrodyellow')
    check = CheckButtons(ax_check, ['Apply Color Mask'], [True])
    check.on_clicked(update)


    # Show the plot
    if not args.auto:
        plt.show()
    # Write the calculation completed file
    with open(calculation_completed_file, 'w') as f:
        f.write("Calculation completed successfully.\n")

    sys.stdout = original_stdout

if __name__ == "__main__":
    main()
