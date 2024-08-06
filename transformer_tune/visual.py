# Data as a list of tuples
data = [
    (1, 1.02),
    (2, 2.72),
    (5, 4.83),
    (10, 11.12),
    (50, 48.09),
    (100, 98.14)
]

import matplotlib.pyplot as plt

# Parse the data
rows = [row for row, time in data]
times = [time for row, time in data]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(rows, times, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Rows')
plt.ylabel('Time (s)')
plt.title('Time vs Number of Rows')
plt.grid(True)

# Save the plot to a file
plt.savefig('plot.png')
print('Plot saved to plot.png')