import os

# --- Configuration ---

# !!! CHANGE THIS VARIABLE TO THE DESIRED STARTING DIRECTORY !!!
# Example: root = "/path/to/your/project" or root = "src"
root = fr"./"

# Extensions to include in the output
extensions = ('.html','.js','.css', '.json', '.md', '.txt','.toml','.ico')

# Directories and patterns to exclude entirely (will not be walked)
# Common examples: virtual environments, git files, caches, build artifacts
exclude_dirs = (
    '.git',
    '__pycache__',
    '.pytest_cache',
    'dist',
    'tests',
    'logs'
)

# Specific files to list in the tree but exclude their *content* from the output
exclude_files = (
    'project_content.txt',
    'write_content_to_file.py',
    'session.json',
    'cm_logo.ico'
)

# Output file name
output_filename = "project_content.txt"

# --- Script Logic ---

# Normalize the root path and make it absolute for reliable comparisons
root = os.path.abspath(root)

print(f"Starting directory: {root}")
print(f"Output file: {output_filename}")

try:
    with open(output_filename, "w", encoding="utf-8") as outfile:
        # === Add Directory Structure ===
        outfile.write("=== Project Directory Structure ===\n")
        outfile.write(f"Root: {root}\n")
        outfile.write("Relevant files and folders (excluding specified patterns):\n\n")

        # Use a single walk for tree structure
        # Keep track of levels for indentation
        start_level = root.count(os.sep)
        for current_root, dirs, files in os.walk(root, topdown=True):
            # Filter directories *in place* to prevent walking into excluded ones
            # Also exclude hidden directories starting with '.'
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

            rel_path_from_start = os.path.relpath(current_root, root)
            level = current_root.count(os.sep) - start_level

            # --- Branch Exclusion Check (for structure) ---
            # If the current directory itself (relative to root) matches an exclusion, skip it.
            # This prevents printing the excluded dir name itself.
            # Note: We already modified `dirs[:]` above, this is for printing the *current* dir.
            # Skip if we are inside a hidden directory (relative path contains '/.')
            # or if a direct component of the relative path is in exclude_dirs
            if rel_path_from_start != '.':
                 path_components = os.path.normpath(rel_path_from_start).split(os.sep)
                 if any(comp in exclude_dirs or comp.startswith('.') for comp in path_components):
                     # We already pruned `dirs` list, so just don't print this entry
                     continue

                 indent = '│   ' * (level -1) + '├── ' if level > 0 else ''
                 outfile.write(f"{indent}{os.path.basename(current_root)}/\n")
            else:
                 # Indicate the root for clarity, even if empty or just containing files
                 outfile.write(".\n")


            # --- File Output for Tree ---
            file_indent = '│   ' * level + '├── '
            files.sort() # Sort files for consistent order
            for file in files:
                # Exclude hidden files starting with '.'
                if file.endswith(extensions) and not file.startswith('.'):
                     outfile.write(f"{file_indent}{file}\n")

        outfile.write("\n\n=== File Contents ===\n\n")

        # === Add File Contents ===
        for current_root, dirs, files in os.walk(root, topdown=True):
            # Apply the same directory filtering as above
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]

            # --- Branch Exclusion Check (for content) ---
            # Similar check to skip processing files in excluded/hidden directories
            rel_path_from_start = os.path.relpath(current_root, root)
            if rel_path_from_start != '.':
                 path_components = os.path.normpath(rel_path_from_start).split(os.sep)
                 if any(comp in exclude_dirs or comp.startswith('.') for comp in path_components):
                     continue # Skip files in this directory


            files.sort() # Sort files for consistent order
            for file in files:
                # Exclude hidden files starting with '.'
                if file.endswith(extensions) and not file.startswith('.'):
                    file_path = os.path.join(current_root, file)
                    # *** KEY CHANGE: Calculate path relative to the specified root ***
                    relative_path = os.path.relpath(file_path, root)

                    # Use OS-agnostic separator for display
                    display_path = relative_path.replace(os.sep, '/')
                    outfile.write(f"=== {display_path} ===\n")

                    if file in exclude_files:
                        outfile.write("--- CONTENT EXCLUDED (listed in exclude_files) ---\n")
                    else:
                        try:
                            # Attempt to read with utf-8, fallback to latin-1 for binary/other files
                            try:
                                with open(file_path, "r", encoding="utf-8") as infile:
                                    outfile.write(infile.read())
                            except UnicodeDecodeError:
                                try:
                                    with open(file_path, "r", encoding="latin-1") as infile:
                                        outfile.write(infile.read())
                                    outfile.write("\n--- (Warning: Read using latin-1 encoding) ---\n")
                                except Exception as inner_e:
                                     outfile.write(f"--- Error reading file (fallback failed): {inner_e} ---\n")

                        except Exception as e:
                            outfile.write(f"--- Error reading file: {e} ---\n")
                    outfile.write("\n\n") # Add separation between file contents
        
    print("Successfully generated project content file.")

except Exception as e:
    print(f"An error occurred: {e}")