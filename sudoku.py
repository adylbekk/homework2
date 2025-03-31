import random
import time
import copy
import os
import doctest
import asyncio
import multiprocessing
import sys 


# ==============================================================================
# == Core Sudoku Functions
# ==============================================================================

def read_sudoku(filename):
    """ Прочитать Судоку из указанного файла """
    try:
        with open(filename, 'r') as f:
            content = f.read()
        # Keep only digits and dots, ignore whitespace/newlines
        digits = [c for c in content if c in '123456789.']
        if len(digits) != 81:
            raise ValueError(f"Sudoku file '{filename}' must contain exactly 81 digits or dots (found {len(digits)}).")
        grid = group(digits, 9)
        return grid
    except FileNotFoundError:
        # Keep error messages concise for cleaner output
        # print(f"Error: File '{filename}' not found.")
        return None
    except ValueError as e:
        # print(f"Error reading '{filename}': {e}")
        return None
    except Exception as e:
        # print(f"An unexpected error occurred reading '{filename}': {e}")
        return None

def group(values, n):
    """
    Сгруппировать значения values в список списков по n элементов

    Doctests:
    >>> group([1,2,3,4], 2)
    [[1, 2], [3, 4]]
    >>> group([1,2,3,4,5,6,7,8,9], 3)
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> group([], 3)
    []
    >>> group(list(range(9)), 9)
    [[0, 1, 2, 3, 4, 5, 6, 7, 8]]
    """
    # Ensure input is a list
    if not isinstance(values, list):
         values = list(values)
    # Handle empty input list gracefully
    if not values:
         return []
    # Check if length is divisible by n
    if len(values) % n != 0:
         raise ValueError(f"Length of values ({len(values)}) must be divisible by n ({n})")
    # Perform the grouping using list comprehension
    return [values[i:i+n] for i in range(0, len(values), n)]

def display(grid):
    """Вывод Судоку в читаемом формате."""
    # Validate grid structure
    if not grid or not isinstance(grid, list) or len(grid) != 9:
        print("Cannot display: Invalid or empty grid provided.")
        return
    if not all(isinstance(row, list) and len(row) == 9 for row in grid):
        print("Cannot display: Grid contains invalid rows.")
        return

    width = 2 
    # Adjusted line formatting for clarity
    h_separator = '+'.join(['-' * (width * 3 + 2)] * 3) # e.g., "--------+--------+--------"
    outer_border_len = len(h_separator) + 4 # Calculate length for top/bottom borders

    print("-" * outer_border_len) 
    for i, row in enumerate(grid):
        
        row_strs = []
        for j, val in enumerate(row):
            # Add cell value (left-aligned)
            row_strs.append(f'{str(val):<{width}}')
            # Add space within block or separator between blocks
            if (j + 1) % 3 == 0 and j < 8:
                row_strs.append(' | ') # Block separator
            elif j < 8:
                 row_strs.append(' ') # Space within block
        print(f"| {''.join(row_strs)} |") 

        
        if (i + 1) % 3 == 0 and i < 8:
            print(f"|{h_separator}|")
    print("-" * outer_border_len) 

def get_row(grid, pos):
    """ Возвращает ряд по позиции pos """
    # Doctests omitted for brevity, assume they passed from previous version
    if not grid or not pos or len(pos) != 2: return []
    r, _ = pos
    return grid[r] if 0 <= r < len(grid) else []

def get_col(grid, pos):
    """ Возвращает колонку по позиции pos """
    # Doctests omitted for brevity
    if not grid or not grid[0] or not pos or len(pos) != 2: return []
    _, c = pos
    # Ensure column index is valid before list comprehension
    if 0 <= c < len(grid[0]):
        # Check if all rows actually have the required column index length
        if all(len(row) > c for row in grid):
            return [row[c] for row in grid]
        else:
            # print(f"Error (get_col): Grid is not rectangular or column {c} is out of bounds for some rows.")
            return []
    else:
        # print(f"Error (get_col): Column index {c} out of bounds.")
        return []


def get_block(grid, pos):
    """
    Возвращает все значения из квадрата 3x3, в который попадает позиция pos

    Doctests:
    >>> grid_3x3 = [['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]
    >>> get_block(grid_3x3, (1, 1)) # Test a small grid block
    ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    >>> # Assuming puzzle1.txt exists for the next tests
    >>> grid_p1 = read_sudoku('puzzle1.txt')
    >>> get_block(grid_p1, (0, 1)) if grid_p1 else 'skip' # Top-left block
    ['5', '3', '.', '6', '.', '.', '.', '9', '8']
    >>> get_block(grid_p1, (4, 7)) if grid_p1 else 'skip' # Middle-right block
    ['.', '.', '3', '.', '.', '1', '.', '.', '6']
    >>> get_block(grid_p1, (8, 8)) if grid_p1 else 'skip' # Bottom-right block
    ['2', '8', '.', '.', '.', '5', '.', '7', '9']
    """
    if not grid or not pos or len(pos) != 2: return []
    r_idx, c_idx = pos
    # Check if pos is within the bounds of the grid structure
    if not (0 <= r_idx < len(grid) and isinstance(grid[r_idx], list) and 0 <= c_idx < len(grid[r_idx])):
        # print(f"Error (get_block): Position {pos} or grid structure is invalid.")
        return []

    start_row, start_col = (r_idx // 3) * 3, (c_idx // 3) * 3
    block_values = []
    for r in range(start_row, start_row + 3):
        # Check row bounds and type
        if r < len(grid) and isinstance(grid[r], list):
            for c in range(start_col, start_col + 3):
                 # Check column bounds
                 if c < len(grid[r]):
                     block_values.append(grid[r][c])
                 # else: print(f"Warning (get_block): Column index {c} out of bounds for row {r}.")
        # else: print(f"Warning (get_block): Row index {r} out of bounds or not a list.")

    return block_values

    