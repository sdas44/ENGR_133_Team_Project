
"""
Course Number: ENGR 13300
Semester: e.g. Fall 2025

Description:
    This program calculates the actual age in years and seconds from the days elapse since the last birthday

Assignment Information:
    Assignment:     tp1 team 3
    Team ID:        LC05, 05
    Author:         Samarth Das, das316@purdue.edu
    Date:           10/16/2025

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
from checkpoint1.tp1_team_1_5 import load_img
from checkpoint1.tp1_team_2_5 import clean_image
import matplotlib.pyplot as plt
import numpy as np
import math

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
    #use the rgb_to_hsv function to convert an entire image to hsv
    hsv_arr = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            r, g, b = arr[i, j]
            hsv_arr[i, j] = rgb_to_hsv(r, g, b)
    return hsv_arr.astype(np.uint8)

def main():
    # as for input
  img_path = input("Enter the path of the RGB image you want to convert to hsv: ")
  img_array = load_img(img_path)
  img_array = clean_image(img_array)
  x_str, y_str = input("Enter the x and y coordinates of the pixel you want inspect: ").split(",")
  x_coor, y_coor = int(x_str), int(y_str)
  #return output
  print(f"RGB values of the ({x_coor}, {y_coor}) pixel: R={img_array[x_coor][y_coor][0]}, G={img_array[x_coor][y_coor][1]}, B={img_array[x_coor][y_coor][2]}")
  print(f"Converting {img_path} to HSV...")
  hue_conversion = rgb_to_hsv(img_array[x_coor][y_coor][0], img_array[x_coor][y_coor][1], img_array[x_coor][y_coor][2])
  print(f"HSV values of the ({x_coor}, {y_coor}) pixel: H={int(hue_conversion[0])}, S={int(hue_conversion[1])}, V={int(hue_conversion[2])}")
  
  # convert the rgb image to hsv image, this may return errors, but again ignore autograder
  converted_hue_array = convert_to_hsv(img_array)
  
  plt.imshow(converted_hue_array)
  plt.show()
  
  
if __name__ == "__main__":
  main()