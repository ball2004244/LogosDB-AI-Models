from rich import print

'''
This file simulates the time taken to process a given number of rows.

Current work is quadratic function
'''

# n = int(input('Enter the number of rows: '))
# n = 100000 # Test for 100,000 rows
n = 1000000 # Test for 1,000,000 rows

# Read coefficients from file
coef = []
coef_file = 'quadratic_coefficients.txt'
with open(coef_file, 'r') as file:
    for line in file:
        coef = list(map(float, line.split()))

# Ensure coefficients are correctly read
a, b, c = coef

# Calculate time in seconds
sec = a * n**2 + b * n + c

# Convert time to hours and days
hours = sec / 3600
days = sec / 86400

# Print the formatted output
print(f'Time taken to process {n:,} rows: {sec:,.2f} seconds ~ {hours:,.2f} hours ~ {days:,.2f} days')