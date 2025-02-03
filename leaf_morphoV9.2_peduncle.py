# Standard library imports
import os
import multiprocessing as mp
import argparse
import traceback
import re
from itertools import combinations
import time
import concurrent.futures

# Third-party libraries
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import tempfile
from pyzbar.pyzbar import decode
from roboflow import Roboflow
import astropy.units as u
from fil_finder import FilFinder2D
from tqdm import tqdm
from scipy.spatial import ConvexHull

# Local application/library specific imports
from plantcv import plantcv as pcv
from qreader import QReader

# Define functions
# Order markers in clockwise order, starting with top left
def order_points_clockwise(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right+most points from the sorted
    # x+roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y+coordinates so we can grab the top+left and bottom+left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now, sort the right-most coordinates according to their
    # y-coordinates so we can grab the top-right and bottom-right
    # points, respectively
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (tr, br) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="int32")


    return process_fil_finder(skeleton, mask)

# Perspective transform function
def perspective_transform(image, corners):
    def order_corner_points(corners):
        # Convert to numpy array for easier manipulation
        corners = np.array(corners, dtype="float32").squeeze()

        # Initialize a list of coordinates that will be ordered
        rect = np.zeros((4, 2), dtype="float32")

        # The top-left point will have the smallest sum, whereas the bottom-right point will have the largest sum
        s = corners.sum(axis=1)
        rect[0] = corners[np.argmin(s)]  # Top-left has the smallest sum
        rect[2] = corners[np.argmax(s)]  # Bottom-right has the largest sum

        # The top-right point will have the smallest difference, whereas the bottom-left will have the largest difference
        diff = np.diff(corners, axis=1)
        rect[1] = corners[np.argmin(diff)]  # Top-right has the smallest difference
        rect[3] = corners[np.argmax(diff)]  # Bottom-left has the largest difference
        top_l, top_r, bottom_r, bottom_l = rect[0], rect[1], rect[2], rect[3]
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between 
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between 
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in 
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))

# find the farthest white pixels from eachother in skeleton
def find_farthest_points(skeleton):
    # Find the white pixels in the skeleton
    white_pixels = np.argwhere(skeleton == 255)

    # If there are fewer than 2 white pixels, return None
    if len(white_pixels) < 2:
        return None

    # Compute the convex hull of the white pixels
    hull = ConvexHull(white_pixels)
    hull_points = white_pixels[hull.vertices]

    # Find the farthest pair of points on the convex hull
    max_distance = 0
    farthest_points = None
    for i in range(len(hull_points)):
        for j in range(i + 1, len(hull_points)):
            distance = np.linalg.norm(hull_points[i] - hull_points[j])
            if distance > max_distance:
                max_distance = distance
                farthest_points = (hull_points[i], hull_points[j])

    return farthest_points

# Leaf image process function
def leaf_morpho(input_image, output_dir):
    error_message = None
    try:
        # Get the file name
        file_name = os.path.basename(input_image)
        file_name = os.path.splitext(file_name)[0]
        try:
            # Find the line name
            pattern = r'^(.*?)(?=_p)'
            match = re.search(pattern, file_name)
            line_name = match.group(1)
        except Exception as e:
            # Regex pattern to extract everything between "picture_" and "_2024"
            pattern = r'picture_(.*?)_2024'
            # Find the match
            match = re.search(pattern, file_name)
            line_name = match.group(1)

        # Read image
        img = cv2.imread(input_image, cv2.IMREAD_COLOR)

        # Call roboflow api
        rf = Roboflow(api_key='l6XPyOniqM4Ecq129cpf')
        
        # Set marker diameter/Users/aja294/Documents/Grape_local/grape_test
        width = 10.5
        height = 9.5
        dia = 0.5

        # Rotate the image
        rotated_img = img

        # Perform instance segmentation to find the markers
        project = rf.workspace().project("morphometric_segmentation")
        model = project.version("3").model

        # Create a temp file on which to run the model
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, rotated_img)

        # Run instance segmentation
        results = model.predict(temp_file_path, confidence=50).json()

        # iterate through the marker centers
        diameters = []
        coordinates = []
        markers = []

        # Iterate through the predictions to find the bounding box for the class "Marker"
        for prediction in results['predictions']:
            if prediction['class'] == 'Marker':
                # Find the center of the markers
                x_center = int(prediction['x'])
                y_center = int(prediction['y'])
                coordinates.append((x_center, y_center))
                # Find the diameter of the markers
                diameter = int(max(prediction['width'], prediction['height']))
                diameters.append(diameter)
                # Find the bounding box of the markers
                x0 = int(prediction['x'] - prediction['width'] / 2)
                y0 = int(prediction['y'] - prediction['height'] / 2)
                x1 = int(prediction['x'] + prediction['width'] / 2)
                y1 = int(prediction['y'] + prediction['height'] / 2)
                markers.append((x0, y0, x1, y1))
                # Convert each marker to white
                points = np.array([[p['x'], p['y']] for p in prediction['points']], dtype=np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(rotated_img, [points], color=(255, 255, 255))

        coordinates_array = np.array(coordinates)
        diameters_array = np.array(diameters)
        mean_dia = np.mean(diameters_array)

      # If coordinates array is greater than four coordinates, find the four that form the corners of a square
        if len(coordinates_array) > 4:
            min_variance = float('inf')
            best_combination = None

            # Iterate through all combinations of four points
            for combination in combinations(coordinates_array, 4):
                combination = np.array(combination)
                # Calculate the pairwise distances
                dists = [np.linalg.norm(combination[i] - combination[j]) for i in range(4) for j in range(i + 1, 4)]
                variance = np.var(dists)
                if variance < min_variance:
                    min_variance = variance
                    corner_points = combination
        else:
            corner_points = np.array([[int(x), int(y)] for corner in coordinates_array for x, y in [corner.ravel()]])
        
        ordered_corner_points = order_points_clockwise(corner_points)
        mask = np.zeros(rotated_img.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [ordered_corner_points], (255,255,255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)
        if len(approx) == 4:
            target_box = perspective_transform(rotated_img, approx)
        else :
            raise ValueError("ERROR: Could not find 4 corners for" + file_name)
        
        # repeat the above pix per in but for cm using a conversion factor
        pix_per_mm_dia = (mean_dia / float(dia)) / 25.4
        pix_per_mm_height = (target_box.shape[0] / float(height)) / 25.4
        pix_per_mm_width = (target_box.shape[1] / float(width)) / 25.4
        pix_per_mm_avg = (pix_per_mm_height + pix_per_mm_width) / 2

        # Run the model to find the peduncle
        project = rf.workspace().project("penduncle-segment-hpx1v")
        model = project.version("1").model

        # Create a temp file on which to run the model
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_file_path = temp_file.name
            cv2.imwrite(temp_file_path, target_box)

        # Run instance segmentation for the peduncle
        results = model.predict(temp_file_path, confidence=10).json()

        # Test if peduncle was found
        if len(results['predictions']) == 0:
            error_message = "Peduncle not found"
            peduncle_mm = 0
            euclidean_peduncle_mm = 0
        else:
            # Find the points of the peduncle
            for prediction in results['predictions']:
                if prediction['class'] == 'peduncle':
                    points = np.array([[p['x'], p['y']] for p in prediction['points']], dtype=np.int32).reshape((-1, 1, 2))

            # Create a mask with the same dimensions as the image
            mask = np.zeros(target_box.shape[:2], dtype=np.uint8)

            # Draw the contours on the mask around the peduncle on the mask
            cv2.drawContours(mask, [points], -1, (255), thickness=cv2.FILLED)

            # Skeletonize the mask the peduncle
            peduncle_skeleton = cv2.ximgproc.thinning(mask)

            # Find the number of pixels in the skeleton
            num_skeleton_pixels = np.count_nonzero(peduncle_skeleton)

            # Convert pixels to mm
            peduncle_mm = num_skeleton_pixels/pix_per_mm_height

            # Find Euclidian distance of peduncle
            ped_points = find_farthest_points(peduncle_skeleton)

            # Find Euclidean distance between farthest points of the peduncle skeleton
            euclidean_distance = np.linalg.norm(ped_points[0] - ped_points[1])

            # Convert the pixels to mm to find peduncle eudlidean distance
            euclidean_peduncle_mm = euclidean_distance / pix_per_mm_avg

        # Convert the target_box to grayscalee
        gray_img = cv2.cvtColor(target_box, cv2.COLOR_BGR2GRAY)

        # Filter out pixels with values greater than 200, these are peduncle
        filtered_pixels = gray_img[(gray_img > 0) & (gray_img <= 200)]

        # Calculate the average of the filtered pixel values, these are the template pixels
        average_value = np.mean(filtered_pixels)

        # Thresh the image at the average pixel value of the template to remove background
        _, thresh = cv2.threshold(gray_img, average_value, 255, cv2.THRESH_BINARY)

        # Get the dimensions of the target box
        height_img, width_img = gray_img.shape[:2]

        # Create a new image with the same dimensions as target box and set all pixel values to the average_value
        average_image = np.full((height_img, width_img), average_value, dtype=np.uint8)

        # Perform image subtraction to get the object image in abscence of shadows
        object_image = cv2.subtract(average_image, gray_img)

        # Create a mask from all pixels in object_image to remove shadows
        mask = cv2.inRange(object_image, 50, 255)

        # Apply mask to cropped image
        masked_img1 = cv2.bitwise_and(target_box, target_box, mask=mask)

        # Find the largest contour 
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)

        # Fill in all other contours with black
        # Create a copy from mask1
        mask = np.zeros_like(mask)

        # Find the largest object in mask2, draw contours around it
        cv2.drawContours(mask, [largest_contour], -1, 255, -1)

        # Change all black pixels to white that are within the contours of the largest object
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Mask masked_iamge with mask2
        masked_img2 = cv2.bitwise_and(masked_img1, masked_img1, mask=mask)

        # Copy image to draw polylines on
        save_img = masked_img2.copy()

        # Draw polylines
        for prediction in results['predictions']:
            if prediction['class'] == 'peduncle':
                points = np.array([[p['x'], p['y']] for p in prediction['points']], dtype=np.int32).reshape((-1, 1, 2))

                # Draw the polyline on the image (example image)
                cv2.polylines(save_img, [points], isClosed=True, color=(0, 0, 255), thickness=5)

        # Skeletonize bincrop
        skeleton = cv2.ximgproc.thinning(mask)
        
        # Process the skeleton using FilFinder2D
        fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=mask)
        fil.preprocess_image(flatten_percent=85)
        fil.create_mask(border_masking=True, verbose=False, use_existing_mask=True)
        fil.medskel(verbose=False)
        fil.analyze_skeletons(branch_thresh=40* u.pix, skel_thresh=10 * u.pix, prune_criteria='length', max_prune_iter=5)
   
        # Save the identified rachis
        rachis_skeleton = fil.skeleton_longpath

        # Find the number of white pixels in skeleton3
        num_white_pixels = np.count_nonzero(rachis_skeleton)
        total_len_mm = num_white_pixels/pix_per_mm_avg

        # Thresh full skeleton for input into farthest point function
        _, rachis_skeleton = cv2.threshold(rachis_skeleton, 0, 255, cv2.THRESH_BINARY)

        # Find farthest points in the rachis skeleton to calculate the total length
        farthest_points = find_farthest_points(rachis_skeleton)

        # Find the Euclidean distance between the farthest points
        euclidean_distance = np.linalg.norm(farthest_points[0] - farthest_points[1])
        euclidean_total_mm = euclidean_distance / pix_per_mm_avg

        # Find the number of white pixels in skeleton3
        num_white_pixels = np.count_nonzero(rachis_skeleton)

        total_len_mm = num_white_pixels/pix_per_mm_avg

        # Fiund farthest points in skel3 
        full_points = find_farthest_points(rachis_skeleton)

        # Find Euclidean distance between farthest points
        euclidean_distance = np.linalg.norm(full_points[0] - full_points[1])
        euclidean_distance_mm = euclidean_distance/pix_per_mm_avg

        if output_dir is not False:
            # Create a new figure for the rachis skeleton image
            fig, ax = plt.subplots()
            # Display the skeleton image
            ax.imshow(skeleton, cmap='gray')
            # Overlay the longest path contour
            ax.contour(rachis_skeleton, colors='r')
            if peduncle_mm != 0:
                # Overlay the peduncle skeleton
                ax.contour(peduncle_skeleton, colors='b')
            # Make the skeleton bolder
            kernel = np.ones((3,3),np.uint8)
            bold_skeleton = cv2.dilate(skeleton, kernel, iterations=5)
            # Display the bold skeleton image
            ax.imshow(bold_skeleton, cmap='gray', alpha=0.5)
            # Remove the axis
            ax.axis('off')
            # Save the figure
            img_name = file_name + '_skel_full.jpg'
            output_path = os.path.join(output_dir, img_name)
            plt.savefig(output_path)

        if error_message is None:
            error_message = "No error"
        source = 0
        data = {
            'line_name': [line_name],
            'total_len_mm': [total_len_mm],
            'euclidian_total_mm': [euclidean_total_mm],
            'peduncle_mm': [peduncle_mm],
            'euclidian_peduncle_mm': [euclidean_peduncle_mm],
            'report': [error_message],
            'source': [0]
        }

        df = pd.DataFrame(data)
        return df
      
    except Exception as e:
        error_message = str(e)
        tb = traceback.extract_tb(e.__traceback__)
        source = {
            'file': tb[-1].filename,
            'line': tb[-1].lineno,
            'code': tb[-1].line
        }
        data = {
            'line_name': [line_name],
            'total_len_mm': ['error'],
            'euclidian_total_mm': ['error'],
            'peduncle_mm': ['error'],
            'euclidian_peduncle_mm': ['error'],
            'report': [error_message],
            'source': [source]
        }

        df = pd.DataFrame(data)
        return df
    
def get_input_images(input_dir):
    try:
        # Ensure the input directory exists
        if not os.path.isdir(input_dir):
            raise ValueError(f"The directory {input_dir} does not exist or is not a directory.")
        else:
            # List all files in the directory
            input_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
            return input_images
    except Exception as e:
        return []
        
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Analyze leaf morphometrics from images")
    parser.add_argument('-i', '--input_dir', type=str, default=None, help='Path to input directory of images to be analyzed.')
    parser.add_argument('-o', '--output_dir', default=None, help='Path to output directory for resulting images.')
    parser.add_argument('-r', '--results_path', default=None, help='Desired path and file name for results.')
    parser.add_argument('-w','--workers', default=None, type=int, help='Number of worker processes to use, if nothing is specified half of all available workers will be used.')
    args = parser.parse_args()
    
    if args.input_dir is not None:
        input_dir = args.input_dir
    else:
        while True:
            input_dir = input("\nPlease enter the path to the the input images to be analyzed: ")
            if os.path.exists(input_dir):
                args.input_dir = input_dir
                break
            else:
                print("Invalid path, please try again.")

    input_images = get_input_images(input_dir)

    if args.output_dir is not None:
        print("\nResulting ouput images will be wrote to:", args.output_dir)
    else:
        while True:
            output_dir = input("\nWould you like to save all output images (WARNING: Resulting file may be large)?\n(y/n): ")
            if output_dir == 'y':
                while True:
                    #ask user if they have a directory or would like to make one
                    output_dir = input("\nWould you like to: (a) Save the images to a known directory? or (b) Make a new folder in the current directory?\n(a/b): ")
                    if output_dir == 'a':
                        while True:
                            # Ask user the pathname to the directory
                            output_dir = input("\nPlease enter the path to the the known image output directory for resulting images: ")
                            if os.path.isdir(output_dir):
                                print("\nIntended output directory confirmed:", output_dir)
                                break
                            else:
                                print("\nIntended output directory not found, please try again.")
                                continue
                    elif output_dir == 'b':
                            output_dir = input("\nPlease enter the desired name of the new image output directory: ")
                            if output_dir != '':
                            # if name was provided do the following loop
                                os.mkdir(output_dir)
                                #get path of new output_dir
                                output_dir = os.path.join(os.getcwd(), output_dir)
                                print("\nNew directory created, images will be saved to:", output_dir)
                                break
                    else:
                        print("Invalid input, please try again.")
                        continue
                    break
                break
            elif output_dir == 'n':
                print("\nNo output directory for images designated resulting images will not be saved.") 
                output_dir = False
                break
            else:
                print("Invalid input, please try again.")
                continue

    if args.results_path is not None:
        print("\nResults will be wrote to:", args.results_path)
        results_path = args.results_path
    else:
        print("\nResults will be saved in a csv in the current directory. To save to a different directory, follow the python call with -r my/intended/directory/myResults.csv.")
        while True:
            results_path = input("\nWould you like the resulting csv to be named using the default result file name - 'leaf_morpho_results.csv'?\n(y/n): ")
            if results_path == 'y':
                results_path = os.path.join(os.getcwd(), 'leaf_morpho_results.csv')
                break
            elif results_path == 'n':
                results_path = input("\nPlease enter the desired name of the results file: ")
                if results_path.endswith('.csv'):
                    results_path = os.path.join(os.getcwd(), results_path)
                    break
                else:
                    results_path = results_path + '.csv'
                    results_path = os.path.join(os.getcwd(), results_path)
                    break
            else:
                print("Invalid input, please try again.")
                continue
            break
   
    if args.workers is not None:
    # Use the value of the --workers argument
        num_workers = args.workers
        print(f'\nUsing  {num_workers} workers for parrallel processing.')
    else:
        # Use a default value
        num_workers = int(mp.cpu_count()/2)
        print("\nUsing half of all available workers for parrallel processing, to change this, enter arguement -w myDesiredNumberOfWorkers following the python call.")
        
    # Create a progress bar
       # Create a progress bar
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks to the executor
        futures = [executor.submit(leaf_morpho, input_image, output_dir) for input_image in input_images]

        # Get the results from the futures
        results = []
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc='Processing images'):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Exception occurred: {e}")

        # Combine all results from leaf_morpho into a pandas dataframe
        results = pd.concat(results, ignore_index=True)

    # convert results to a csv file
    results.to_csv(results_path, index=True)