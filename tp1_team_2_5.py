
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
from PIL import Image

def image_to_array(image_path):
  """Converts an image to a numpy array."""
  img = Image.open(image_path)
  img_array = np.array(img)
  return img_array

def string_to_binary(sequence):
  """Convert a string to its binary representation."""
  binary_string = ""
  for char in sequence:
    binary_string += format(ord(char), '08b') #08b mean 8 bit binary, ord(char) gives ascii value for the character
    
  return binary_string

def normalize_array(arr):
  """Normalize a numpy array to the range [0, 255]"""
  arr_min = np.min(arr)
  arr_max = np.max(arr)
  normalized_arr = (arr - arr_min) / (arr_max - arr_min) * 255
  return normalized_arr.astype(np.uint8)

def find_binary_message(img_array, start_sequence, end_sequence):
  """Find the binary message hidden in the image array."""
  binary_message = ""
  if img_array.ndim == 2:  # Grayscale image
    rows, cols = img_array.shape
    for c in range(cols):
      for r in range(rows):
        pixel_value = img_array[r, c]
        lsb = pixel_value & 1  # Get the least significant bit
        binary_message += str(lsb)
  else:  # Color image
    rows, cols, channels = img_array.shape
    for c in range(cols):
      for r in range(rows):
        for ch in range(channels):
          pixel_value = img_array[r, c, ch]
          lsb = pixel_value & 1  # Get the least significant bit
          binary_message += str(lsb)
        
  # check if the start sequence and end sequence are in the binary message
  start_index = binary_message.find(start_sequence)
  end_index = binary_message.find(end_sequence, start_index + len(start_sequence))
  if start_index != -1 and end_index != -1:
    return binary_message[start_index + len(start_sequence):end_index]
  else:
    return "Start or end sequence not found in the image."

def main():
  image = pathlib.Path(input("Enter the path of the image you want to load: "))
  img_array = image_to_array(image)
  img_array = normalize_array(img_array)
  start_sequence = input("Enter the start sequence: ")
  end_sequence = input("Enter the end sequence: ")
  binary_start_sequence = string_to_binary(start_sequence)
  binary_end_sequence = string_to_binary(end_sequence)
  binary_message = find_binary_message(img_array, binary_start_sequence, binary_end_sequence)
  print("Extracted Message: ", binary_message)

if __name__ == "__main__":
    main()