import numpy as np
import matplotlib.pyplot as plt

'''
This code visualizes the relationship between the number of rows and the time taken to process them for both single and multi-processing modes.
'''

# File paths
TIME_FILE = 'time_vs_rows.txt'
MP_TIME_FILE = 'time_vs_rows_mp.txt'
MP_CONFIG_FILE = 'mp_config.txt'
SINGLE_COEF_FILE = 'cpu-coef.txt'
MULTI_COEF_FILE = 'cpu-coef-mp.txt'
PLOT_FILE = 'single_vs_multi.png'

def read_data(file_path):
    """Read data from a file and return it as a list of tuples."""
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                row, time = line.split()
                data.append((int(row), float(time)))
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
    except ValueError:
        print(f"Error: Incorrect format in {file_path}.")
    return data

def read_config(file_path):
    """Read configuration from a file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return "Number of CPU cores: 32\nMultiprocessing start method: fork"

def fit_quadratic(x, y):
    """Fit a quadratic polynomial to the data and return the coefficients."""
    return np.polyfit(x, y, 2)

def store_coefficients(file_path, coefficients):
    """Store the coefficients in a file."""
    with open(file_path, 'w') as file:
        file.write(' '.join(map(str, coefficients)))

def plot_data(single_rows, single_times, single_fit, multi_rows, multi_times, multi_fit, config):
    """Plot the actual times and fitted polynomial for both single and multi-processing."""
    plt.figure(figsize=(10, 6))
    plt.plot(single_rows, single_times, marker='o', linestyle='-', color='b', label='Single-processing Actual')
    plt.plot(single_rows, single_fit, linestyle='--', color='c', label='Single-processing Fit')
    plt.plot(multi_rows, multi_times, marker='o', linestyle='-', color='g', label='Multi-processing Actual')
    plt.plot(multi_rows, multi_fit, linestyle='--', color='y', label='Multi-processing Fit')
    plt.xlabel('Number of Rows')
    plt.ylabel('Time (s)')
    plt.title('Time vs Number of Rows (Single vs Multi-processing)')
    plt.grid(True)

    # Add text annotations for each point
    for row, time in zip(single_rows, single_times):
        plt.annotate(f'({row}, {time:.2f})', (row, time), textcoords="offset points", xytext=(0, 10), ha='center')
    for row, time in zip(multi_rows, multi_times):
        plt.annotate(f'({row}, {time:.2f})', (row, time), textcoords="offset points", xytext=(0, 10), ha='center')

    # Add multiprocessing configuration as text annotation
    plt.annotate(config, xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, ha='left', va='top', bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

    # Save the plot to a file
    plt.legend()
    plt.savefig(PLOT_FILE)
    print(f'Plot saved to {PLOT_FILE}')

def main():
    # Read data
    single_data = read_data(TIME_FILE)
    if not single_data:
        print(f"No data found in {TIME_FILE}. Exiting.")
        return

    multi_data = read_data(MP_TIME_FILE)
    if not multi_data:
        print(f"No data found in {MP_TIME_FILE}. Exiting.")
        return

    # Parse the data
    single_rows, single_times = zip(*single_data)
    multi_rows, multi_times = zip(*multi_data)

    # Fit quadratic polynomial and store coefficients
    single_coef = fit_quadratic(single_rows, single_times)
    store_coefficients(SINGLE_COEF_FILE, single_coef)

    multi_coef = fit_quadratic(multi_rows, multi_times)
    store_coefficients(MULTI_COEF_FILE, multi_coef)

    # Generate fitted values
    single_fit = np.polyval(single_coef, single_rows)
    multi_fit = np.polyval(multi_coef, multi_rows)

    # Read multiprocessing configuration
    config = read_config(MP_CONFIG_FILE)

    # Plot the data
    plot_data(single_rows, single_times, single_fit, multi_rows, multi_times, multi_fit, config)

if __name__ == '__main__':
    main()