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
    