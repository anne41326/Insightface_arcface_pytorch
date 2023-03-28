"Create pairs.txt for testing accuracy of model"
from pathlib import Path
import glob
import os.path
import numpy as np
import os

THIS_FILE_PATH = Path(__file__)
VALDATA_PATH = THIS_FILE_PATH.parent / "data/val"
PAIRS_TXT_PATH = THIS_FILE_PATH.parent / "pairs.txt"


def create_match_content(INPUT_DATA=VALDATA_PATH) -> set:
    """
    Desc:
    Create matched pairs content by path of val folder and photos of it

    Args：
    INPUT_DATA: path of val folder(photos for testing)
    
    Output: None
    
    """
    matched_result = set()
    k = 0
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]

    while len(matched_result) < 3000:
        for sub_dir in sub_dirs[1:]:

            # to get vaild photos in the current dir
            extensions = 'jpg'
            # get photos in list
            file_list = []
            # use os.path.basename(sub_dir)to return the name of sub_dir
            dir_name = os.path.basename(sub_dir)
            # print(dir_name)
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extensions)
            # use glob.glob(file_glob)to get all photos in specific folder, then put in file_list
            file_list.extend(glob.glob(file_glob))
            if not file_list: continue
            if len(file_list) >= 2:

                # use names of dirs to get names of lables
                label_name = dir_name

                length = len(file_list)

                # randomly get pairs of name to base_name1 and base_name2
                random_number1 = np.random.randint(length)
                random_number2 = np.random.randint(length)
                while random_number1 == random_number2:
                    random_number1 = np.random.randint(length)
                    random_number2 = np.random.randint(length)
                base_name1 = os.path.basename(file_list[random_number1 %
                                                        length])
                base_name2 = os.path.basename(file_list[random_number2 %
                                                        length])

                if (file_list[random_number1 % length] !=
                        file_list[random_number2 % length]):
                    # put outcome in dic,1 stand for the same person image(matched)
                    matched_result.add(label_name + '\\' + base_name1 + ' ' +
                                       label_name + '\\' + base_name2 + ' 1')
                    k = k + 1
    return matched_result, k


def create_unmatch_content(INPUT_DATA=VALDATA_PATH) -> set:
    """
    Desc:
    Create unmatched pairs content by path of val folder and photos of it

    Args：
    INPUT_DATA: path of val folder(photos for testing)
    
    Output: None
    
    """

    unmatched_result = set()
    k = 0
    # create class names
    while len(unmatched_result) < 3000:
        sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
        length_of_dir = len(sub_dirs)
        for j in range(24):

            for i in range(1, length_of_dir):
                class1 = sub_dirs[i]
                random_number = np.random.randint(length_of_dir)
                while random_number == 0 | random_number == i:
                    random_number = np.random.randint(length_of_dir)
                class2 = sub_dirs[random_number]
                class1_name = os.path.basename(class1)
                class2_name = os.path.basename(class2)

                # to get vaild photos in the current dir
                extensions = 'jpg'
                file_list1 = []
                file_list2 = []

                # get photos in list
                file_glob1 = os.path.join(INPUT_DATA, class1_name,
                                          '*.' + extensions)
                file_list1.extend(glob.glob(file_glob1))
                file_glob2 = os.path.join(INPUT_DATA, class2_name,
                                          '*.' + extensions)
                file_list2.extend(glob.glob(file_glob2))

                # get folders name,then put them on base_name1,2
                if file_list1 and file_list2:
                    base_name1 = os.path.basename(file_list1[j %
                                                             len(file_list1)])
                    base_name2 = os.path.basename(file_list2[j %
                                                             len(file_list2)])
                    s = class2_name + '\\' + base_name2 + ' ' + class1_name + '\\' + base_name1 + ' 0'

                    # exclude photos of identical people
                    if (s not in unmatched_result):
                        unmatched_result.add(class1_name + '\\' + base_name1 +
                                             ' ' + class2_name + '\\' +
                                             base_name2 + ' 0')
                        if len(unmatched_result) > 3000:
                            break
                    k = k + 1
    return unmatched_result, k


def create_pairs(INPUT_DATA=VALDATA_PATH):
    """
    Desc:
    Create pairs.txt by matched and unmacthed pairs

    Args：
    INPUT_DATA: path of val folder(photos for testing)
    
    Output: .txt file(pairs.txt)
    
    """

    result, k1 = create_match_content(INPUT_DATA=VALDATA_PATH)

    #print(k1)

    result_un, k2 = create_unmatch_content(INPUT_DATA=VALDATA_PATH)
    #print(k2)
    file = open(PAIRS_TXT_PATH, 'w',
                encoding="utf-8")  #add utf-8 to read chinese name file
    result1 = list(result)
    result2 = list(result_un)
    #file.write('10 300\n')
    for i in range(10):
        for pair in result1[i * 300:i * 300 + 300]:

            file.write(pair + '\n')

        for index, pair in enumerate(result2[i * 300:i * 300 + 300]):
            #print(len(result2[i*300:i*300+300])) 300
            #print(index, len(result2[i*300:i*300+300])-1, pair)
            if i < 9:
                file.write(pair + '\n')
            else:
                file.write(pair +
                           ('\n' if index < len(result2[i * 300:i * 300 +
                                                        300]) - 1 else ''))
    file.close()


if __name__ == '__main__':
    THIS_FILE_PATH = Path(__file__)
    VALDATA_PATH = THIS_FILE_PATH.parent / "data/val"
    create_pairs(INPUT_DATA=VALDATA_PATH)
