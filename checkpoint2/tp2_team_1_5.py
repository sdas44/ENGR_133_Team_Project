
"""
Course Number: ENGR 13300
Semester: e.g. Fall 2025

Description:
    This program calculates the actual age in years and seconds from the days elapse since the last birthday

Assignment Information:
    Assignment:     tp2 team 1
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

def load_img(image_path):
  """Converts an image to a numpy array."""
  img = Image.open(image_path)
  img_array = np.array(img)
  img_array = linearization_array(img_array) #0-1 normalization already exists within the function
  img_array = img_array * 255
  if img_array.shape[-1]==4:
    return img_array[:, :, :3].astype(np.uint8)
  return img_array.astype(np.uint8)

def linearization_array(arr):
  # linearize the values based on the conditions of the normalized_array_1
  arr = arr / 255.0
  
  #if array is grayscale then loop through 2 dimensions
  if arr.ndim == 2:
    for i in range(arr.shape[0]):
      for j in range(arr.shape[1]):
        if arr[i][j] <= 0.04045:
          arr[i][j] = arr[i][j] / 12.92
        else:
          arr[i][j] = ((arr[i][j] + 0.055) / 1.055) ** 2.4
    return arr
  # loop through all pixels
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      for ch in range(arr.shape[2]):
        if arr[i][j][ch] <= 0.04045:
          arr[i][j][ch] = arr[i][j][ch] / 12.92
        else:
          arr[i][j][ch] = ((arr[i][j][ch] + 0.055) / 1.055) ** 2.4
  return arr

def gaussian_filter(arr, std):
  # first we must pad the image
  padded_image = np.pad(arr, ((3,3),(3,3)), mode='constant', constant_values=0)
  kernel = np.zeros((7, 7))
  center = 7 // 2
  
  # create the kernel
  for x in range(7):
    for y in range(7):
        dx = x - center
        dy = y - center
        kernel[x, y] = (1 / (2 * np.pi * std**2)) * np.exp(-(dx**2 + dy**2) / (2 * std**2))
        
  #normalize the kernel
  kernel /= np.sum(kernel)
  
  #create the output array
  output_arr = np.zeros_like(arr)
  
  # apply gausian kernel to image
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      region = padded_image[i:i+7, j:j+7]
      output_arr[i,j] = np.sum(region * kernel)
  return output_arr.astype(float) #TODO: Needs to be changed if I resubmit this assignment to the autograder

def rgb_to_grayscale(arr):
  # converts rgb to grayscale
  gry_arry = np.zeros((arr.shape[0], arr.shape[1]))
  for i in range(arr.shape[0]):
    for j in range(arr.shape[1]):
      gry_arry[i][j] = int(0.2126 * arr[i][j][0] + 0.7152 * arr[i][j][1] + 0.0722 * arr[i][j][2])
      
  return gry_arry.astype(np.uint8)

def main():
  STD = 1
  image = pathlib.Path(input("Enter the path of the image file: "))
  img_array = load_img(image)
  gry_img_array = rgb_to_grayscale(img_array)
  
  gaussian_array = gaussian_filter(gry_img_array, STD)
  print(gaussian_array)
  plt.imshow(gaussian_array, cmap='gray')
  plt.show()

if __name__ == "__main__":
  main()