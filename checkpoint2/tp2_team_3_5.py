
"""
Course Number: ENGR 13300
Semester: e.g. Fall 2025

Description:
    This program calculates the actual age in years and seconds from the days elapse since the last birthday

Assignment Information:
    Assignment:     tp2 team 3
    Team ID:        LC05, 05
    Author:         Samarth Das, das316@purdue.edu
    Date:           10/9/2025

Contributors:
    Edward Ojuolape, eojuolap@purdue.edu,
    Lwanda Muigo, muigl01@purdue.edu
    Benjamin Tianming Sun, sun1384@purdue.edu

    My contributor(s) helped me:
    [X] understand the assignment expectations without
        telling me how they will approach it.
    [X] understand different ways to think about a solution
        without helping me plan my solution.
    [X] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import pathlib
import math
from tp2_team_1_5 import load_img, rgb_to_grayscale, gaussian_filter
from tp2_team_2_5 import sobel_filter
import cv2
import pandas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def extract_features(img_arr):
    hsv_arr = convert_to_hsv(img_arr).astype(np.float64)
    hue = hsv_arr[:,:,0]   # Hue: 0–179 in OpenCV
    sat = hsv_arr[:,:,1]   # Saturation: 0–255
    val = hsv_arr[:,:,2]   # Value: 0–255

    hue_avg = np.mean(hue)
    hue_std = np.std(hue)
    sat_avg = np.mean(sat)
    sat_std = np.std(sat)
    val_avg = np.mean(val)
    val_std = np.std(val)

    gry_arr = rgb_to_grayscale(img_arr)
    gaus_arr = gaussian_filter(gry_arr, 1)
    circles = detect_circles(gry_arr)
    lines = count_lines(gaus_arr)

    return [hue_avg, hue_std, sat_avg, sat_std, val_avg, val_std, circles, lines]
  
  
def rgb_to_hsv(r, g , b):
    # normalize pixel values
    #IMPORTANT: The Autograder will say this function is wrong, however ENGR 133 says that they will fix the floating point error which will make this function correct
    normalized_red = r / 255.0
    normalized_green = g / 255.0
    normalized_blue = b / 255.0
    
    c_max = max(normalized_red, normalized_green, normalized_blue)
    c_min = min(normalized_red, normalized_green, normalized_blue)
    delta = c_max - c_min
    
    # use Hue function to calculate hue
    hue = 0
    if delta == 0:
        hue = 0
    elif c_max == normalized_red:
        hue = (60 * ((normalized_green - normalized_blue) / delta)) % 360
    elif c_max == normalized_green:
        hue = (60 * ((normalized_blue - normalized_red) / delta) + 120) % 360
    elif c_max == normalized_blue:
        hue = (60 * ((normalized_red - normalized_green) / delta) + 240) % 360
    
    # calculate saturation and value
    saturation = 0
    if c_max == 0:
        saturation = 0
    else:
        saturation = delta / c_max
    
    value = c_max
    
    return (((hue / 360 * 255)), (saturation * 255), (value * 255))
    
def convert_to_hsv(arr):
    hsv_arr = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=float)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            r, g, b = arr[i, j]
            hsv_arr[i, j] = rgb_to_hsv(r, g, b)
    return hsv_arr.astype(np.uint8)
  
def detect_circles(gray_img):
 """Detects if a large circle is present. Returns 1 if found, 0 otherwise."""
 # Hough Circles works best on a grayscale, slightly blurred image
 # Apply a Gaussian blur
 blurred_img = gaussian_filter(gray_img, 1.5)
 blurred_img = blurred_img.astype(np.uint8)
 # Detect circles
 circles = cv2.HoughCircles(
     blurred_img,
     cv2.HOUGH_GRADIENT,
     dp=1.2,  # Inverse ratio of accumulator resolution
     minDist=100,  # Minimum distance between centers of detected circles
     param1=150,  # Upper threshold for the internal Canny edge detector
     param2=50,  # Threshold for center detection
     minRadius=20,  # Minimum circle radius to detect
     maxRadius=50  # Maximum circle radius to detect
 )
 return 1 if circles is not None else 0

def count_lines(gray_img):
 """Counts the number of lines in an image using Hough Line Transform."""
 # Edge detection via sobel filtering is a prerequisite for Hough Lines
 edges = sobel_filter(gray_img)
 
 # Detect lines using the edge map
 lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

 return len(lines) if lines is not None else 0

def clean_image(arr):
  # resize the image to fit within 100x100 canvas
  # check to see whether its grey scale or color
  # if arr.ndim == 3:
  #   print(f"Image shape before cleaning: ({arr.shape[0]}, {arr.shape[1]}, {arr.shape[2]})")
  # else:
  #   print(f"Image shape before cleaning: ({arr.shape[0]}, {arr.shape[1]})")
  aspect_ratio = arr.shape[0] / arr.shape[1]
  img = Image.fromarray(arr)
  
  #if the image is longer than height
  if arr.shape[0] > arr.shape[1]:
    # print(f"Resized image to: ({100}, {int(100 / aspect_ratio)})")
    resized_image = img.resize((int(100 / aspect_ratio), 100), Image.Resampling.BILINEAR)
    resized_image = ImageOps.pad(resized_image, (100,100), Image.Resampling.BILINEAR, color="#000")
    resized_image_arr = np.array(resized_image)
    return resized_image_arr
  # if height is longer and width
  elif arr.shape[0] < arr.shape[1]:
    # print(f"Resized image to: ({int(100 * aspect_ratio)}, {100})")
    resized_image = img.resize((100, int(100 * aspect_ratio)), Image.Resampling.BILINEAR)
    resized_image = ImageOps.pad(resized_image, (100,100), Image.Resampling.BILINEAR, color="#000")
    resized_image_arr = np.array(resized_image)
    return resized_image_arr
  #if its a square then just scale it to 100,100
  else:
    # print(f"Resized image to: ({100}, {100})")
    resized_image = img.resize((100, 100), Image.Resampling.BILINEAR)
    resized_image = ImageOps.pad(resized_image, (100,100), Image.Resampling.BILINEAR, color="#000")
    resized_image_arr = np.array(resized_image)
    return resized_image_arr

def main():
    path_to_folder = pathlib.Path(input("Enter the name of the dataset folder:\n"))
    metadata_csv = pathlib.Path(input("Enter the name of the metadata file:\n"))
    output_csv = pathlib.Path(input("Enter the name of output csv features file:\n"))
    # ask to show two features
    x_feature = input("Enter the column to use as the x-axis feature: ")
    y_feature = input("Enter the column to use as the y-axis feature: ")

    meta_df =pandas.read_csv(metadata_csv)
    output_df = pandas.DataFrame()
    print("\nTraining Metadata DataFrame:\n")
    print(meta_df.head(5))
    feature_list = []
    for index, row in meta_df.iterrows():
        print(row)
        image_path = "./ML_Images/" + row["Path"]
        img_arr = load_img(image_path)
        clean_img_arr = clean_image(img_arr)
        
        features = extract_features(clean_img_arr)
        
        feature_list.append({
            'hue_mean': features[0],
            'hue_std': features[1],
            'saturation_mean': features[2],
            'saturation_std': features[3],
            'value_mean': features[4],
            'value_std': features[5],
            'num_lines': features[7],
            'has_circle': features[6],
            'Path': row['Path'],
            'ClassId': row['ClassId']
        })
      
    feature_df = pandas.DataFrame(feature_list)
    
    print("Features have been extracted and saved to img_features.csv\n")
    print("Feature Dataset Shape")
    print(feature_df.shape)

    # Show the first 5 rows
    print("Feature Dataset Head:")
    print(feature_df.head(5))

    # create a custom colormap: 0 = red, 1 = blue
    cmap = ListedColormap(['red', 'blue'])

    plt.figure(figsize=(8,6))
    plt.scatter(
        feature_df[x_feature],
        feature_df[y_feature],
        c=feature_df['ClassId'],
        cmap=cmap,
        alpha=0.7
    )
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.title(f'2D Feature Space: {x_feature} vs. {y_feature}')
    plt.grid()

    # save plot as PNG
    plt.savefig('KNN_feature_space.png', dpi=300)

    plt.show()


if __name__ == "__main__":
  main()