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

def find_empty_positions(grid):
    """
    Найти первую свободную позицию (ячейку с '.') в пазле.
    Возвращает кортеж (row, col) или None, если пустых нет.

    Doctests:
    >>> find_empty_positions([['1', '2', '.'], ['4', '5', '6'], ['7', '8', '9']])
    (0, 2)
    >>> find_empty_positions([['1', '2', '3'], ['4', '.', '6'], ['7', '8', '9']])
    (1, 1)
    >>> find_empty_positions([['1', '2', '3'], ['4', '5', '6'], ['.', '8', '9']])
    (2, 0)
    >>> find_empty_positions([['1', '2', '3'], ['4', '5', '6'], ['7', '8', '9']]) is None
    True
    """
    if not grid or not isinstance(grid, list): return None # Handle invalid grid
    for r, row in enumerate(grid):
        # Check if row is a list before trying index()
        if isinstance(row, list):
            try:
                # Find the index of the first '.' in the current row
                c = row.index('.')
                return (r, c)
            except ValueError:
                # '.' not found in this row, continue to the next
                continue
    return None # No empty cells found in any row

def find_possible_values(grid, pos):
    """
    Вернуть множество всех возможных цифр ('1'-'9') для указанной позиции.

    Doctests:
    >>> grid_p1 = read_sudoku('puzzle1.txt') # Assumes puzzle1.txt exists
    >>> values_02 = find_possible_values(grid_p1, (0, 2)) if grid_p1 else set()
    >>> sorted(list(values_02))
    ['1', '2', '4']
    >>> values_47 = find_possible_values(grid_p1, (4, 7)) if grid_p1 else set()
    >>> sorted(list(values_47))
    ['2', '5', '9']
    """
    if not grid or not pos or len(pos) != 2: return set()
    r_idx, c_idx = pos
    # Check if pos is within the bounds of the grid structure
    if not (0 <= r_idx < len(grid) and isinstance(grid[r_idx], list) and 0 <= c_idx < len(grid[r_idx])):
        # print(f"Error (find_possible_values): Position {pos} or grid structure invalid.")
        return set()

    all_digits = set('123456789')

    # Get values from row, column, and block using helper functions
    # Use set union for combining values
    used_digits = (
        set(get_row(grid, pos)) |
        set(get_col(grid, pos)) |
        set(get_block(grid, pos))
    )

    # The possible values are the ones not used (after removing '.')
    return all_digits - (used_digits - {'.'})

def solve(grid):
    """
    Решение пазла Судоку (рекурсивный бэктрекинг). Модифицирует grid НА МЕСТЕ.
    Возвращает решенный grid или False, если решения нет.

    Doctests:
    >>> grid_p1 = read_sudoku('puzzle1.txt') # Assumes puzzle1.txt exists
    >>> if grid_p1: grid_copy = copy.deepcopy(grid_p1); solution = solve(grid_copy); result = (solution is not False)
    >>> result if grid_p1 else 'skip' # Check if a solution was found
    True
    >>> solution[0] if grid_p1 else 'skip' # Check first row of known solution
    ['5', '3', '4', '6', '7', '8', '9', '1', '2']
    >>> # Test unsolvable (modify puzzle1 slightly)
    >>> if grid_p1: grid_unsolvable = read_sudoku('puzzle1.txt'); grid_unsolvable[0][1] = '6'; result_unsolvable = (solve(grid_unsolvable) is False)
    >>> result_unsolvable if grid_p1 else 'skip'
    True
    """
    # Find the next empty cell
    empty_pos = find_empty_positions(grid)

    # Base case: If no empty positions, the puzzle is solved
    if not empty_pos:
        # Optional final check (can slow down significantly if used)
        # return grid if check_solution(grid) else False
        return grid # Assume validity maintained by recursion

    r, c = empty_pos

    # Try filling the empty cell with possible values
    # Sorting the values ensures deterministic behavior if multiple solutions exist
    # (though Sudoku puzzles should ideally have unique solutions)
    for value in sorted(list(find_possible_values(grid, empty_pos))):
        grid[r][c] = value # Place the value (attempt)

        # Recursively try to solve the rest of the puzzle
        if solve(grid): # Recursive call returns solved grid or False
            # If the recursive call found a solution, propagate it back up
            return grid

        # Backtrack: If the value didn't lead to a solution, reset the cell
        grid[r][c] = '.'

    # If no possible value worked for this empty cell, trigger backtracking
    return False # Indicate failure for this path

def check_solution(solution):
    """
    Проверяет валидность решённого Судоку
    (Все ячейки '1'-'9', без повторов в рядах, колонках, блоках).

    Doctests:
    >>> grid_p1 = read_sudoku('puzzle1.txt') # Assumes puzzle1.txt exists
    >>> solved_grid = solve(copy.deepcopy(grid_p1)) if grid_p1 else None
    >>> check_solution(solved_grid) if solved_grid else False
    True
    >>> if solved_grid: solved_grid[0][0] = '.'; result_incomplete = check_solution(solved_grid)
    >>> result_incomplete if solved_grid else 'skip'
    False
    >>> if solved_grid: solved_grid[0][0] = solved_grid[0][1]; result_duplicate = check_solution(solved_grid) # Create duplicate in row 0
    >>> result_duplicate if solved_grid else 'skip'
    False
    """
    # Basic structural checks first
    if not solution or not isinstance(solution, list) or len(solution) != 9 \
       or not all(isinstance(row, list) and len(row) == 9 for row in solution):
        return False

    expected = set('123456789')

    
    if not all(set(row) == expected for row in solution):
        
        return False

    
    for c in range(9):
        
        if set(solution[r][c] for r in range(9)) != expected:
            # print(f"Check failed: Column {c} constraint violation.")
            return False

    
    for block_r_start in range(0, 9, 3): 
        for block_c_start in range(0, 9, 3): 
            
            block = set(solution[r][c]
                        for r in range(block_r_start, block_r_start + 3)
                        for c in range(block_c_start, block_c_start + 3))
            if block != expected:
                 # print(f"Check failed: Block starting at ({block_r_start}, {block_c_start}) constraint violation.")
                 return False

    # If all checks pass
    return True

def generate_sudoku(N):
    """
    Генерация судоку с N >= 0 заполненными ячейками.
    Гарантирует РЕШАЕМОСТЬ, но не УНИКАЛЬНОСТЬ решения.

    Doctests:
    >>> random.seed(42) # for predictable doctest output
    >>> grid_40 = generate_sudoku(40)
    >>> isinstance(grid_40, list) # Check return type
    True
    >>> sum(1 for row in grid_40 for e in row if e == '.') # 81 - 40 = 41 empty cells
    41
    >>> solution_40 = solve(copy.deepcopy(grid_40)) # Solve a copy to check validity
    >>> check_solution(solution_40)
    True
    >>> grid_81 = generate_sudoku(81) # N = 81 should result in a full grid
    >>> sum(1 for row in grid_81 for e in row if e == '.')
    0
    >>> check_solution(grid_81) # A generated full grid should be valid
    True
    >>> grid_0 = generate_sudoku(0) # N = 0 should result in an empty grid
    >>> sum(1 for row in grid_0 for e in row if e == '.')
    81
    >>> solution_0 = solve(copy.deepcopy(grid_0)) # Solving an empty grid should work
    >>> check_solution(solution_0)
    True
    """
    
    base_grid = [['.' for _ in range(9)] for _ in range(9)]

    
    
    solved_grid = solve(copy.deepcopy(base_grid))

    
    if not solved_grid or not check_solution(solved_grid):
        print("Error (generate_sudoku): Failed to create a valid base solution.")
        return None

    
    N = max(0, min(N, 81))

    
    cells_to_remove = 81 - N
    if cells_to_remove == 0:
        return solved_grid # Return the fully solved grid if N=81

    #Get all possible (row, col) positions and shuffle them randomly.
    all_positions = [(r, c) for r in range(9) for c in range(9)]
    random.shuffle(all_positions)

    #Create the puzzle grid by copying the solved grid.
    puzzle_grid = copy.deepcopy(solved_grid)

    #Remove cells (set to '.') until the desired number is removed.
    removed_count = 0
    positions_to_try = list(all_positions) # Use a copy to pop from

    # Basic removal method - does NOT guarantee a unique solution.
    while removed_count < cells_to_remove and positions_to_try:
        r, c = positions_to_try.pop()
        # Only remove if the cell is not already empty (safeguard)
        if puzzle_grid[r][c] != '.':
             puzzle_grid[r][c] = '.'
             removed_count += 1

    

    return puzzle_grid

# ==============================================================================
# == Helper Functions for Concurrent/Parallel Execution
# ==============================================================================

# --- Function for Asyncio ---
async def run_solve_async(fname):
    """Async wrapper: Reads, displays original, solves (using executor),
       displays solved, times, prints result."""
    # Using a simple separator for less visual clutter
    separator = f"--- Async Task for {fname} ---"
    print(f"\n{separator}")
    start_read_time = time.time()
    grid = read_sudoku(fname)
    read_time = time.time() - start_read_time
    if not grid:
        print(f"Failed to read {fname} (took {read_time:.4f}s).")
        print("-" * len(separator))
        return

    print(f"Read {fname} successfully (took {read_time:.4f}s).")
    print("Original Puzzle:")
    display(grid) # Display original puzzle

    print(f"Solving {fname}...")
    start_solve_time = time.time()
    loop = asyncio.get_running_loop()
    solved_grid = None
    original_grid_copy = copy.deepcopy(grid) # Keep original for display if solve fails

    try:
        # Run the SYNCHRONOUS solve function in the default thread pool executor
        # Pass a DEEP COPY of the grid because 'solve' modifies it in-place
        solved_grid = await loop.run_in_executor(None, solve, copy.deepcopy(grid))
    except Exception as e:
        print(f"Error solving {fname} in executor: {e}")

    end_solve_time = time.time()
    solve_time = end_solve_time - start_solve_time

    result_prefix = f"RESULT for {fname}:"

    if solved_grid and check_solution(solved_grid):
        print(f"Solved Puzzle ({solve_time:.4f} seconds):")
        display(solved_grid) # Display solved puzzle
        print(f"{result_prefix} ✅ SOLVED")
    elif solved_grid: # Returned a grid but it's invalid
        print(f"Attempted Solution ({solve_time:.4f} seconds) - INVALID:")
        display(solved_grid) # Display the invalid result
        print(f"{result_prefix} ❌ INVALID SOLUTION")
    else: # solve returned False or exception occurred
        print(f"Original Puzzle ({solve_time:.4f} seconds) - UNSOLVED:")
        display(original_grid_copy) # Show the original again
        print(f"{result_prefix} ❌ FAILED TO SOLVE")
    print("-" * len(separator))


# --- Function for Multiprocessing (ACCEPTS LOCK) ---
def run_solve_sync_timed(fname, print_lock): # Added print_lock parameter
    """Synchronous function for MP: Reads, solves, times, prints result using a lock."""
    # This code runs in a SEPARATE process.
    process_id = os.getpid()
    # Reading and solving happen outside the lock to maximize parallelism

    start_read_time = time.time()
    grid = read_sudoku(fname)
    read_time = time.time() - start_read_time
    original_grid_copy = copy.deepcopy(grid) # Keep original separately

    # --- Print initial info and original grid (protected by lock) ---
    with print_lock:
        separator = f"--- [MP Process {process_id}] Processing {fname} ---"
        print(f"\n{separator}") # Start with newline
        if not grid:
            print(f"Failed to read {fname} (took {read_time:.4f}s).")
            print("-" * len(separator))
            return # Exit this process early if read failed

        print(f"Read {fname} successfully (took {read_time:.4f}s).")
        print(f"Original Puzzle:")
        try:
            display(grid) # Display original puzzle under lock
        except Exception as e:
            print(f"Error displaying original: {e}")
        print(f"Solving {fname}...") # Print solving message inside lock before releasing

    # --- Solving happens OUTSIDE the lock ---
    start_solve_time = time.time()
    solved_grid = None
    try:
        # Solve a DEEP COPY - crucial for separate process memory
        # 'solve' is the original synchronous function
        solved_grid = solve(copy.deepcopy(grid))
    except Exception as e:
         # Print error immediately (might interleave, but better than silent failure)
         # Consider acquiring lock even for error print if strict ordering needed
         print(f"\n[MP Process {process_id}] Error during solving {fname}: {e}\n")

    end_solve_time = time.time()
    solve_time = end_solve_time - start_solve_time

    # --- Print results (protected by lock) ---
    with print_lock:
        result_prefix = f"[MP Process {process_id}] RESULT for {fname}:"
        # Use the same separator length basis for consistency
        separator = f"--- [MP Process {process_id}] Finished {fname} ---"

        if solved_grid and check_solution(solved_grid):
             print(f"[MP Process {process_id}] Solved Puzzle ({solve_time:.4f} seconds):")
             try:
                 display(solved_grid) # Display solved puzzle under lock
             except Exception as e:
                 print(f"Error displaying solved: {e}")
             print(f"{result_prefix} ✅ SOLVED")

        elif solved_grid: # Invalid solution
             print(f"[MP Process {process_id}] Attempted Solution ({solve_time:.4f} seconds) - INVALID:")
             try:
                 display(solved_grid) # Display invalid result under lock
             except Exception as e:
                 print(f"Error displaying invalid solution: {e}")
             print(f"{result_prefix} ❌ INVALID SOLUTION")

        else: # Failed to solve
             print(f"[MP Process {process_id}] Original Puzzle ({solve_time:.4f} seconds) - UNSOLVED:")
             try:
                 # Display the original grid copy we saved earlier
                 display(original_grid_copy)
             except Exception as e:
                 print(f"Error displaying original (unsolved): {e}")
             print(f"{result_prefix} ❌ FAILED TO SOLVE")
        print("-" * len(separator)) # Final separator for this file's output block


# ==============================================================================
# == Main Execution Block
# ==============================================================================
if __name__ == '__main__':



    # --- Basic Setup ---
    puzzle_files = ('puzzle1.txt', 'puzzle2.txt', 'puzzle3.txt')
    print("=" * 70)
    print(" Sudoku Solver Lab Execution")
    print("=" * 70)

    # --- Check for Puzzle Files ---
    print("--- Checking Puzzle Files ---")
    files_exist = True
    for fname in puzzle_files:
        if not os.path.exists(fname):
            print(f"WARNING: Puzzle file '{fname}' not found.")
            files_exist = False
        else:
            print(f"- Found '{fname}'.")
    if not files_exist:
        print("\nERROR: One or more required puzzle files are missing.")
        print("Please create them (e.g., using the puzzle file generator script) or upload them.")
        print("Solver tests will be skipped.")
    print("-" * 70)


    # --- Run Doctests ---
    print("\n--- Running Doctests ---")
    # Ensure puzzle1.txt exists for doctests that depend on it
    doctest_puzzle1_needed = True # Flag if doctests rely on puzzle1.txt
    if not os.path.exists('puzzle1.txt') and doctest_puzzle1_needed:
        print("Attempting to create dummy 'puzzle1.txt' for doctests...")
        try:
            # Content for dummy puzzle1.txt
            dummy_content = ("53..7....\n6..195...\n.98....6.\n8...6...3\n"
                             "4..8.3..1\n7...2...6\n.6....28.\n...419..5\n....8..79\n")
            with open('puzzle1.txt', 'w') as f: f.write(dummy_content)
            print("Dummy 'puzzle1.txt' created successfully.")
            # If files were initially missing, re-check now that one is created
            if not files_exist:
                 files_exist = all(os.path.exists(f) for f in puzzle_files)
        except IOError as e:
            print(f"Failed to create dummy 'puzzle1.txt': {e}. Some doctests might fail.")

    # Run doctests, capture results
    # Use verbose=True to see detailed output on failures
    (failures, tests) = doctest.testmod(verbose=False)

    if tests == 0:
        print("No doctests found or executed.")
    elif failures == 0:
        print(f"All {tests} executed doctests passed! ✅")
    else:
        # Doctest automatically prints failure details,
        print(f"\n*** {failures} out of {tests} DOCTESTS FAILED! Review errors above. *** ❌")
    print("-" * 70)


    # --- Run Solvers (only if puzzle files were found) ---
    if files_exist:

        # --- Asyncio Execution (Handling Running Loop) ---
        print("\n" + "=" * 30 + " Solving Puzzles using Asyncio " + "=" * 30)
        print("(Concurrent execution via thread pool for CPU-bound solve)")

        async def main_async_runner():
            # Create tasks for each puzzle file
            # Each task calls run_solve_async which handles reading, solving, displaying
            tasks = [asyncio.create_task(run_solve_async(fname)) for fname in puzzle_files]
            await asyncio.gather(*tasks) # Wait for all tasks to complete

        asyncio_start_time = time.time()
        try:
            # Attempt standard asyncio.run first
            asyncio.run(main_async_runner())
        except RuntimeError as e:
            # If asyncio.run fails because a loop is already running (common in notebooks)
            if "cannot be called from a running event loop" in str(e):
                print("\nWARNING: Detected running asyncio event loop.")
                print("Attempting to schedule tasks on existing loop (may not block script completion).")
                # Schedule the runner on the existing loop.
                # Note: In a standard script, this might exit before tasks finish.
                
                try:
                    loop = asyncio.get_event_loop() # Get the existing loop
                    # Ensure future allows waiting if possible, but might not block in all contexts
                    future = asyncio.ensure_future(main_async_runner())
                    
                    
                    print("Asyncio tasks scheduled on existing loop.")
                except Exception as inner_e:
                     print(f"Failed to schedule on existing loop: {inner_e}")
            else:
                # Re-raise other unexpected RuntimeErrors
                print(f"\nAn unexpected RuntimeError occurred during asyncio execution: {e}")
                # raise e # Uncomment to stop execution on other runtime errors
        except Exception as e:
            # Catch other potential exceptions during setup/execution
            print(f"\nAn unexpected error occurred during asyncio setup/execution: {e}")

        asyncio_end_time = time.time()
        # Note: Timing reflects time to launch/handle tasks, might not be total solve time if loop wasn't fully blocked.
        print(f"\nAsyncio - Approx. time for task handling: {asyncio_end_time - asyncio_start_time:.4f} seconds")
        print("-" * 70)


        
        print("\n" + "=" * 25 + " Solving Puzzles using Multiprocessing " + "=" * 25)
        print("(Parallel execution with locked printing for cleaner output)")

        
        print_lock = multiprocessing.Lock()
        processes = []
        multiprocessing_start_time = time.time()

        try:
            for fname in puzzle_files:
                # Pass the lock object as an argument to the target function
                p = multiprocessing.Process(target=run_solve_sync_timed, args=(fname, print_lock))
                processes.append(p)
                p.start() # Start the child process

            # Wait for all child processes to complete their execution
            for p in processes:
                p.join() # Blocks the main process until process 'p' finishes

        except Exception as e:
             print(f"\nAn error occurred during multiprocessing execution: {e}")

        multiprocessing_end_time = time.time()
        print(f"\nMultiprocessing - Total time for all processes: {multiprocessing_end_time - multiprocessing_start_time:.4f} seconds")
        print("-" * 70)

    else:
         # If puzzle files were missing, confirm skipping solver sections
         print("\nSkipping Asyncio and Multiprocessing solver tests due to missing puzzle files.")
         print("-" * 70)


    # --- Generation Example ---
    print("\n--- Sudoku Generation Example ---")
    num_clues = 35 # Example: generate a puzzle with roughly 35 clues
    print(f"Generating a puzzle with ~{num_clues} clues...")
    gen_start_time = time.time()
    generated_puzzle = generate_sudoku(num_clues)
    gen_create_time = time.time() - gen_start_time

    if generated_puzzle:
        print(f"Generated Puzzle (creation took {gen_create_time:.4f}s):")
        display(generated_puzzle)

        print("\nVerifying generated puzzle is solvable...")
        verify_start_time = time.time()
        # Solve a copy to verify without modifying the displayed generated puzzle
        gen_solution = solve(copy.deepcopy(generated_puzzle))
        verify_solve_time = time.time() - verify_start_time

        if gen_solution and check_solution(gen_solution):
             print(f"Generated puzzle successfully solved and verified (solve check took {verify_solve_time:.4f}s). ✅")
             # Optionally display the solution:
             # print("Generated Solution:")
             # display(gen_solution)
        else:
             print(f"ERROR: Generated puzzle could not be solved or verified (check took {verify_solve_time:.4f}s). ❌")
    else:
        print(f"Failed to generate puzzle (took {gen_create_time:.4f}s).")
    print("-" * 70)


    print("\nExecution finished.")
    print("=" * 70)