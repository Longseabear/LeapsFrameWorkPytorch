import numpy as np
import os

# arg -> dataset path
#dataset_path = "/media/it-315/hard/data_256"
#output_path = "../file_list.txt"

def fetch_img_list(dir):
    filenames = os.listdir(dir)
    file_list = []
    for filename in filenames:
        file_path = os.path.join(dir,filename)
        if os.path.isdir(file_path):
            file_list += fetch_img_list(file_path)
        else:
            file_list.append(file_path + "\n")
    return file_list

def fetch_img_filename_list(dir):
    filenames = os.listdir(dir)
    file_list = set()
    for filename in filenames:
        file_path = os.path.join(dir,filename)
        if os.path.isdir(file_path):
            file_list += fetch_img_list(file_path)
        else:
            file_list.add(filename + "\n")
    return file_list

def make_flist(dataset_path, output_path):
    file_list = fetch_img_list(dataset_path)
    print("file list {} ".format(file_list), len(file_list))

    with open(output_path,'w') as f:
        f.writelines(file_list)

def fetch_sceneflow_list(dir, with_label=False):
    filenames = os.listdir(dir)
    file_list = []
    for filename in filenames:
        file_path = os.path.join(dir, filename)

        if filename == 'left' or filename == 'right':

            file_path = os.path.join(dir, 'left')
            file_list_left = fetch_sceneflow_list(file_path)
            file_path = os.path.join(dir, 'right')
            file_list_right = fetch_sceneflow_list(file_path)
            assert len(file_list_left)==len(file_list_right)
            if with_label:
                return [file_list_left[i] + ' ' + file_list_right[i] + ' ' + str((int(file_list_left[i][-14:-4])+1) % 2) + '\n' for i in range(len(file_list_left))]
            return [file_list_left[i] + ' ' + file_list_right[i] +'\n' for i in range(len(file_list_left))]
        elif os.path.isdir(file_path):
            file_list += fetch_sceneflow_list(file_path)
        else:
            file_list.append(file_path)
    return file_list

def make_sceneflow_flist(dataset_path, output_path):
    print('make flist')
    dataset_list = os.listdir(dataset_path)
    total_list = []
    for dataset in dataset_list:
        if dataset.endswith('cleanpass'):
            full_path = os.path.join(dataset_path, dataset)
            total_list += fetch_sceneflow_list(full_path)

    with open(output_path, 'w') as f:
        f.writelines(total_list)

    print('make end')

def make_coarse_to_find_datset(dataset_path):
    print('make flist')
    final_list = []
    for i in range(2600):
        c = dataset_path + "/left_levin/{:05d}.bmp".format(i)
        m = dataset_path + "/right/{:05d}.png".format(i)
        final_list.append(c + " " + m)
    print('make end')
    return final_list

def make_sceneflow_flist_with_labels(dataset_path, output_path):
    print('make flist with label')
    total_list = fetch_sceneflow_list(dataset_path, with_label=True)
    with open(output_path, 'w') as f:
        f.writelines(total_list)
    print('make end')

#/media/cvip/3a659201-b774-4847-9d55-7737a17a5a9e/repo/SimilarityTrainSet