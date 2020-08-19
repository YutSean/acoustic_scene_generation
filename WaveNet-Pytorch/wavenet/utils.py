from os import listdir
from os.path import join, isdir, isfile

def list_files(loc, return_dirs=False, return_files=True, recursive=False, valid_exts=None):
    """
    Copied from: https://github.com/JimBoonie/PythonHelpers

    Return a list of all filenames within a directory loc.

    Inputs:
        loc - Path to directory to list files from.
        return_dirs - If true, returns directory names in loc. (default: False)
        return_files - If true, returns filenames in loc. (default: True)
        recursive - If true, searches directories recursively. (default: False)
        valid_exts - If a list, only returns files with extensions in list. If None,
            does nothing. (default: None)

    Outputs:
        files - List of names of all files and/or directories in loc.
    """
    
    files = [join(loc, x) for x in listdir(loc)]

    if return_dirs or recursive:
        # check if file is directory and add it to output if so
        is_dir = [isdir(x) for x in files]
        found_dirs = [files[x] for x in range(len(files)) if is_dir[x]]
    else:
        found_dirs = []

    if return_files:
        # check if file is not directory and add it to output
        is_file = [isfile(x) for x in files]
        found_files = [files[x] for x in range(len(files)) if is_file[x]]
    else:
        found_files = []

    if recursive and not return_dirs:
        new_dirs = []
    else:
        new_dirs = found_dirs

    deeper_files = []
    if recursive:
        for d in found_dirs:
            deeper_files.extend(list_files(d, 
                                           return_dirs=return_dirs, 
                                           return_files=return_files,
                                           recursive=recursive))

    if isinstance(valid_exts, (list, tuple)):
        concat_files = found_files + deeper_files
        new_files = []
        for e in valid_exts:
            new_files.extend([f for f in concat_files if f.endswith(e)])
    else:
        new_files = found_files + deeper_files

    return new_dirs + new_files
