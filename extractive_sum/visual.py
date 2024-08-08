import matplotlib.pyplot as plt
import numpy as np

'''
This code to visualize the relationship between the number of rows and the time taken to process them.
'''

# Read data from the file
time_file = 'time_vs_rows.txt'
plot_file = 'gpu-plot-v1.png'
data = []
with open(time_file, 'r') as file:
    for line in file:
        row, time = line.split()
        data.append((int(row), float(time)))

# Parse the data
rows, times = zip(*data)

# Fit a quadratic polynomial to the data
coefficients = np.polyfit(rows, times, 2)
a, b, c = coefficients
print(f'Quadratic coefficients: a={a}, b={b}, c={c}')
print(f'Fitted quadratic equation: y = {a:.8f}x^2 + {b:.8f}x + {c:.8f}')

# Save the coefficients to a file
coeff_file = 'gpu-coef-v1.txt'
with open(coeff_file, 'w') as file:
    file.write(f'{a} {b} {c}\n')

# Plot the data
plt.figure(figsize=(10, 6))
plt.plot(rows, times, marker='o', linestyle='-',
         color='b', label='Data points')
plt.xlabel('Number of Rows')
plt.ylabel('Time (s)')
plt.title('Time vs Number of Rows')
plt.grid(True)

# Add text annotations for each point
for row, time in data:
    plt.annotate(f'({row}, {time:.2f})', (row, time),
                 textcoords="offset points", xytext=(0, 10), ha='center')

# Plot the quadratic fit
x_fit = np.linspace(min(rows), max(rows), 100)
y_fit = a * x_fit**2 + b * x_fit + c
plt.plot(x_fit, y_fit, color='r', linestyle='--', label='Quadratic fit')

# Save the plot to a file
plt.legend()
plt.savefig(plot_file)
print(f'Plot saved to {plot_file}')
