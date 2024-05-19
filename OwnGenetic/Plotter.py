import matplotlib.pyplot as plt
import numpy as np

# Define a function to read the data from the file
def read_data(filename):
  """
  This function reads numbers from a text file and stores them in a list.

  Args:
      filename: The path to the text file.

  Returns:
      A list of numbers read from the file.
  """
  data = []
  with open(filename, 'r') as f:
    for line in f:
      data.append(float(line.strip()))
  return data

# Specify the filename containing the numbers
filename = "Epoch"  # Replace with your actual filename

# Read the data from the file
data = read_data(filename)

# Extract x-axis (line numbers) and y-axis (data points)
x = np.arange(len(data))  # Create a list from 0 to length of data-1

# Create the line plot
plt.plot(x, data)

# Add labels and title to the plot
plt.xlabel("Line Number")
plt.ylabel("Data Value")
plt.title("Line Graph of Data from " + filename)

# Show the plot
plt.show()