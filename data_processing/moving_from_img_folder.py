import os
import numpy
import cv2
import shutil
import glob


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
    names = sorted(names)
    paths = sorted(paths)
    return names, paths, folders

def moving(folders):

    for folder in folders:
        for content in os.listdir(folder):
            if content == "img":
                working_path = os.path.join(folder, "img")
                names, paths, folders = get_names(working_path)

                for path in paths:
                    shutil.move(path,folder)

                os.rmdir(os.path.join(folder, "img"))

            if content.endswith(".txt"):
                os.remove(os.path.join(folder,content))

if "__main__" == __name__:

    # Get the root path of the files
    root_path = "/media/atr17/HDD Storage/Datasets_Download/Full_Catheter_Dataset/All_axial_dataset/Masks/Val"

    names, paths, folders = get_names(root_path)
    moving(folders=folders)


    print("Done")