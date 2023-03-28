"Create dataset and rename it "
from pathlib import Path
import os

import splitfolders

THIS_FILE_PATH = Path(__file__)
DATA_PATH = THIS_FILE_PATH.parent / "data"
# data type of DATA_PATH is Pathlib.WindowsPath
# Pathlib.WindowsPath is a data type in Python used to represent paths in the Windows
# operating system. It is one of the classes in the pathlib module that is specifically
# designed for handling paths.
VALDATA_PATH = THIS_FILE_PATH.parent / "data/val"


def spiltrandom_folder(Input_Folder: str = DATA_PATH,
                       output: str = DATA_PATH,
                       seed: int = 1337,
                       ratio: tuple = (.7, .3),
                       group_prefix: int = None,
                       move: bool = False) -> None:
    """Randomly spilt data in the folder into multiple folders with appointed 
    ratio or numbers,for training dataset and testing dataset(optionally valdating 
    dataset), more detailded of spiltfolders, please check 
    https://github.com/jfilter/split-folders"""
    # splitfolders.ratio(Input_Folder=utils.get_relative_path('data'),
    #                    output=Input_Folder,
    #                    seed=1337,
    #                    ratio=(.7, .3),
    #                    group_prefix=None,
    #                    move=False)
    splitfolders.ratio(Input_Folder, output, seed, ratio, group_prefix, move)


def rename_valdata(parent_folder: str = VALDATA_PATH.resolve()) -> None:
    """
    Desc:
    Rename filename of each data  with name of sub folder and numbers 
    in sequence such as  A_01.jpg,A_02.jpg in A folders

    Args: 
    parent_folders: str, the path of the parent folder

    Output: None
    
    """
    #parent_folder = (THIS_FILE_PATH.parent / "data/val").resolve()
    for obj in parent_folder.glob('*'):
        if obj.is_dir():
            count = 0
            for photo in obj.glob('*.jpg'):
                photo.rename(photo.parent /
                             '{0}_{1}.jpg '.format(photo.parent.name, count))
                count += 1


if __name__ == '__main__':
    spiltrandom_folder(Input_Folder=DATA_PATH,
                       output=DATA_PATH,
                       seed=1337,
                       ratio=(.7, .3),
                       group_prefix=None,
                       move=False)

    rename_valdata(parent_folder=(VALDATA_PATH.resolve()))
