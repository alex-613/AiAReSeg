import os

def get_names(root_path):

    """
    Grabs the names of of all of the files under a root path
    """

    #root_path = _join_paths(root_path,seq_no)

    # dir_list = os.listdir(root_path)
    names = []
    paths = []
    folders = []
    for path, subdirs, files in os.walk(root_path):
        for name in files:
            # print(os.path.join(path, name))
            # print(name)
            file_dir = os.path.join(os.path.basename(path),name)
            names.append(file_dir)
            paths.append(os.path.join(path, name))
        for subdir in subdirs:
            if subdir != "img":
                folders.append(os.path.join(root_path, subdir))

    folders = sorted(folders, key=lambda x:x[-3:])
    return names, paths, folders

def renaming(folders,starting_index=51):
    """Grabs the folders and renames it"""
    index = starting_index
    for folder in folders:
        sequence_name = folder.split("/")[-1]
        sequence_path = folder.split("/")[:-1]
        sequence_path = "/".join(sequence_path)
        sequence_no = sequence_name.split("-")[-1]
        index_str = str(index)
        new_sequence_name = os.path.join(sequence_path, f"Catheter-{index_str}")
        new_sequence_path = os.path.join(sequence_path, new_sequence_name)

        os.rename(folder, new_sequence_path)

        index += 1


if "__main__" == __name__:

    # Get the root path of the files
    root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/Masks_1/Val"

    starting_index = 675

    old_names, old_path, old_folders = get_names(root_path)
    renaming(old_folders, starting_index=starting_index)
    print("Done")