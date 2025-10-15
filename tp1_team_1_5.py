
"""
Course Number: ENGR 13300
Semester: e.g. Fall 2025

Description:
    This program calculates the actual age in years and seconds from the days elapse since the last birthday

Assignment Information:
    Assignment:     team project 1 task 1
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

import pathlib
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def image_to_array(image_path):
  """Converts an image to a numpy array."""
  img = Image.open(image_path)
  img_array = np.array(img)
  return img_array

def normalize_array(arr):
  """Normalize a numpy array to the range [0, 255]"""
  arr_min = np.min(arr)
  arr_max = np.max(arr)
  normalized_arr = (arr - arr_min) / (arr_max - arr_min) * 255
  return normalized_arr.astype(np.uint8)

def main():
  image = pathlib.Path(input("Enter the path of the image you want to load: "))
  img_array = image_to_array(image)
  img_array = normalize_array(img_array)
  if len(img_array.shape) == 2:
    plt.imshow(img_array, cmap='gray')
  else:
    plt.imshow(img_array)
  plt.show()
  
if __name__ == "__main__":
    main()