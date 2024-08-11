from rich import print
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

'''
This file simulates the time taken to process a given number of rows.

Current work is quadratic function
'''

# Number of rows to process
# n = 1000000  # Test for 1M rows
n = 20000000   # Test for 20M rows
# n = 1000000000  # Test for 1B rows

def read_coefficients(file_path):
    """Read coefficients from a file."""
    with open(file_path, 'r') as file:
        for line in file:
            return list(map(float, line.split()))

def calculate_time(coefficients, n):
    """Calculate the time using the quadratic function."""
    a, b, c = coefficients
    return a * n**2 + b * n + c

def convert_time(seconds):
    """Convert time from seconds to hours and days."""
    hours = seconds / 3600
    days = seconds / 86400
    return hours, days

def create_results_panel(label, seconds, hours, days, n):
    """Create a panel with the formatted results."""
    table = Table(title=label)
    table.add_column("Metric", justify="right", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    table.add_row("Rows", f"{n:,}")
    table.add_row("Seconds", f"{seconds:,.2f}")
    table.add_row("Hours", f"{hours:,.2f}")
    table.add_row("Days", f"{days:,.2f}")

    return Panel(table)

def read_config(file_path):
    """Read configuration from a file."""
    try:
        with open(file_path, 'r') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return "Number of CPU cores: 32\nMultiprocessing start method: fork"

def main():
    # File paths for coefficients
    single_coef_file = 'cpu-coef.txt'
    multi_coef_file = 'cpu-coef-mp.txt'
    config_file = 'mp_config.txt'

    # Read coefficients
    single_coef = read_coefficients(single_coef_file)
    multi_coef = read_coefficients(multi_coef_file)

    # Calculate time for single-processing
    sec_single = calculate_time(single_coef, n)
    hours_single, days_single = convert_time(sec_single)

    # Calculate time for multi-processing
    sec_multi = calculate_time(multi_coef, n)
    hours_multi, days_multi = convert_time(sec_multi)

    # Create panels for results
    single_panel = create_results_panel('Single-processing', sec_single, hours_single, days_single, n)
    multi_panel = create_results_panel('Multi-processing', sec_multi, hours_multi, days_multi, n)

    # Read configuration
    config = read_config(config_file)
    config_panel = Panel(config, title="Multi-processing Config")

    # Display all panels in a single row
    print(Panel(Columns([single_panel, multi_panel, config_panel]), title="Processing Time Simulation"))

if __name__ == '__main__':
    main()