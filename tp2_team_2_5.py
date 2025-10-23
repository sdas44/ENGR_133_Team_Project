
"""
Course Number: ENGR 13300
Semester: e.g. Fall 2025

Description:
    This program calculates the actual age in years and seconds from the days elapse since the last birthday

Assignment Information:
    Assignment:     tp2 task 1
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
from PIL import Image
import pathlib
import math
from tp2_team_1_5 import load_img, rgb_to_grayscale, gaussian_filter

def sobel_filter(gry_arr):
  # create sobel kernels
  sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=float)
  sobel_y = np.array([
    [-1, -2, -1],
    [0,  0,  0],
    [1,  2,  1]
], dtype=float)
  
  # create the padded image
  padded_image = np.pad(gry_arr, ((1,1),(1,1)), mode='constant', constant_values=0)
  print(padded_image)
  
  # create the gradient x and y to sum later
  gradient_x = np.zeros_like(gry_arr)
  gradient_y = np.zeros_like(gry_arr)
  final_gradient = np.zeros_like(gry_arr)
  
  # apply gausian kernel to image
  for i in range(gry_arr.shape[0]):
    for j in range(gry_arr.shape[1]):
      region = padded_image[i:i+3, j:j+3]
      gradient_x[i,j] = np.sum(region * sobel_x)
      gradient_y[i,j] = np.sum(region * sobel_y)
      # create the final gradient
      final_gradient[i,j] = np.sqrt(gradient_x[i,j]**2 + gradient_y[i,j]**2)
  # clip and threshold
  final_gradient = np.clip(final_gradient, 0, 255)
  final_gradient = np.where(final_gradient >= 50, 255, 0).astype(np.uint8)
  print(final_gradient)
  return final_gradient
  
  
  

def main():
  STD = 1
  image = pathlib.Path(input("Enter the path to the image file: "))
  img_array = load_img(image)
  gry_img_array = rgb_to_grayscale(img_array)
  
  gaussian_array = gaussian_filter(gry_img_array, STD) # in order for this to work, guassian_array must return an np array of type float otherwise it doesnt work
  sobel_array = sobel_filter(gaussian_array)
  
  plt.imshow(sobel_array, cmap='gray')
  plt.show()
  
if __name__ == "__main__":
  main()