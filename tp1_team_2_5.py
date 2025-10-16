
"""
Course Number: ENGR 13300
Semester: e.g. Fall 2025

Description:
    This program grabs the binary message from the image the user provides

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
from PIL import Image, ImageOps
from tp1_team_1_5 import image_to_array
import matplotlib.pyplot as plt

def clean_image(arr):
  # resize the image to fit within 100x100 canvas
  if arr.ndim == 3:
    print(f"Image shape before cleaning: ({arr.shape[0]}, {arr.shape[1]}, {arr.shape[2]-1})")
  else:
    print(f"Image shape before cleaning: ({arr.shape[0]}, {arr.shape[1]})")
  aspect_ratio = arr.shape[0] / arr.shape[1]
  img = Image.fromarray(arr)
  
  if arr.shape[0] > arr.shape[1]:
    if arr.ndim == 3:
      print(f"Resized image to: ({100}, {int(100 / aspect_ratio)}, {arr.shape[2]-1})")
    else:
      print(f"Resized image to: ({100}, {int(100 / aspect_ratio)})")
    resized_image = img.resize((100, 100 / aspect_ratio), Image.Resampling.BILINEAR)
    resized_image = ImageOps.pad(resized_image, (100,100), Image.Resampling.BILINEAR, "#0000")
    resized_image_arr = np.array(resized_image)
    return resized_image_arr
  elif arr.shape[0] < arr.shape[1]:
    if arr.ndim == 3:
      print(f"Resized image to: ({int(100 * aspect_ratio)}, {100}, {arr.shape[2]-1})")
    else:
      print(f"Resized image to: ({int(100 * aspect_ratio)}, {100})")
    resized_image = img.resize((int(100 * aspect_ratio), 100), Image.Resampling.BILINEAR)
    resized_image = ImageOps.pad(resized_image, (100,100), Image.Resampling.BILINEAR, "#0000")
    resized_image_arr = np.array(resized_image)
    return resized_image_arr
  else:
    if arr.ndim == 3:
      print(f"Resized image to: ({100}, {100}, {arr.shape[2]-1})")
    else:
      print(f"Resized image to: ({100}, {100})")
    resized_image = img.resized((100, 100), Image.Resampling.BILINEAR)
    resized_image = ImageOps.pad(resized_image, (100,100), Image.Resampling.BILINEAR, "#0000")
    resized_image_arr = np.array(resized_image)
    return resized_image_arr


def main():
  image = pathlib.Path(input("Enter the path of the image you want to load: "))
  img_array = image_to_array(image)
  img_array = clean_image(img_array)
  
  print("Image shape after cleaning: (100, 100)")
  
  if img_array.ndim == 2:
    plt.imshow(img_array, cmap='gray')
  else:
    plt.imshow(img_array)
  plt.show()
if __name__ == "__main__":
    main()