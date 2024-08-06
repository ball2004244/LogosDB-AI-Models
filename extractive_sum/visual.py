import matplotlib.pyplot as plt

# Read data from the file
data = []
with open('time_vs_rows.txt', 'r') as file:
    for line in file:
        row, time = line.split()
        data.append((int(row), float(time)))

# Parse the data
rows = [row for row, time in data]
times = [time for row, time in data]

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(times, rows, marker='o', linestyle='-', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Number of Rows')
plt.title('Number of Rows vs Time')
plt.grid(True)

# Use logarithmic scale for y-axis to better show the relationship
plt.yscale('log')

# Add text annotations for each point
for row, time in data:
    plt.annotate(f'({row}, {time:.2f})', (time, row), textcoords="offset points", xytext=(0,10), ha='center')

# Save the plot to a file
plt.savefig('plot.png')
print('Plot saved to plot.png')