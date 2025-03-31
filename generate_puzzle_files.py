import os

# Define the puzzle data as multiline strings
# Using .strip() to remove potential leading/trailing blank lines from the triple quotes
puzzle1_content = """
53..7....
6..195...
.98....6.
8...6...3
4..8.3..1
7...2...6
.6....28.
...419..5
....8..79
""".strip()

puzzle2_content = """
1.......2
.4.2.3.1.
..5..6..4
.........
.8.....3.
..6...4..
4.3..8...
.1..5.9.8
.......7.
""".strip()

puzzle3_content = """
.....6.7.
4.5.8.1..
.2.1.....
...5.....
.3.4.2.5.
.....1...
.....8.4.
..6.7.2.8
.9.3.....
""".strip()

# Dictionary mapping filenames to their content
puzzles_to_create = {
    'puzzle1.txt': puzzle1_content,
    'puzzle2.txt': puzzle2_content,
    'puzzle3.txt': puzzle3_content
}

# Loop through the dictionary and create each file
print("Generating puzzle files...")
for filename, content in puzzles_to_create.items():
    try:
        with open(filename, 'w') as f:
            f.write(content)
            # Add a newline at the end if the content doesn't end with one,
            # some read operations might expect it. strip() removed it.
            f.write('\n')
        print(f"Successfully created: {filename}")
    except IOError as e:
        print(f"Error creating file {filename}: {e}")

print("\nPuzzle file generation complete.")

# Optional: Verify files were created (useful in Colab)
print("\nVerifying created files:")
for filename in puzzles_to_create.keys():
    if os.path.exists(filename):
        print(f"- {filename} exists.")
    else:
        print(f"- {filename} NOT FOUND.")
