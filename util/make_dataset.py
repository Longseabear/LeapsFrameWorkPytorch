import numpy as np
import scipy
import random
import skimage.io as io
import torch
import warnings
import torch.nn.functional as F

warnings.filterwarnings(action='ignore')

def luma_difference(x,y):
    return np.sqrt(np.sum((x-y)*(x-y)))
flist_path = './flist'
total_data_num = 10000000

file_list = [filename[:-1] for filename in open(flist_path, 'r').readlines()]
dataset_num = len(file_list)
assign_data_num = total_data_num//dataset_num

max_disp = 192

print(dataset_num, assign_data_num)
valid = [2,4,6]
candidate = list(np.arange(1,10))
print(set(candidate)-set(valid))
new_flist = []
dataset_idx = 7939560

output_path = "/media/cvip/3a659201-b774-4847-9d55-7737a17a5a9e/repo/SimilarityTrainSet"
output_left = output_path + "/left/"
output_right = output_path + "/right/"

random.shuffle(file_list)

# for sample in file_list:
for path in file_list:
    left, right = path.split(' ')
    left = io.imread(left)/255
    right = io.imread(right)/255

    h,w,c = left.shape
    remain_num = assign_data_num
    missing_point = 0
    while remain_num>0:
        dx = random.randint(0,w-max_disp-30)
        dy = random.randint(0,h-30)

        valid = []
        right_patch = right[dy:dy + 30, dx:dx + 30, :]
        for i in range(max_disp):
            left_patch = left[dy:dy + 30, dx + i :dx + 30 + i, :]
            val = np.floor(luma_difference(right_patch, left_patch))
            if val<=2:
                valid.append(i)

        candidate = list(np.arange(0, max_disp))
        candidate = list(set(candidate) - set(valid))
        if len(valid) == 0 or len(candidate) == 0:
            missing_point+=1
            if missing_point>=5000:
                print(missing_point)
                break
            continue
        random.shuffle(candidate)
        random.shuffle(valid)

        left_patch = left[dy:dy + 30, dx + valid[0]:dx + 30 + valid[0], :]
        path_right = output_right + "{0:010d}.png".format(dataset_idx)
        path_left = output_left + "{0:010d}.png".format(dataset_idx)
        io.imsave(path_left,np.uint8(left_patch*255))
        io.imsave(path_right,np.uint8(right_patch*255))
        new_flist.append(path_left + " " + path_right + " 1\n")
        dataset_idx = dataset_idx+1
        remain_num -= 1

        left_patch = left[dy:dy + 30, dx + candidate[0]:dx + 30 + candidate[0], :]
        path_right = output_right + "{0:010d}.png".format(dataset_idx)
        path_left = output_left + "{0:010d}.png".format(dataset_idx)
        io.imsave(path_left,np.uint8(left_patch*255))
        io.imsave(path_right,np.uint8(right_patch*255))
        new_flist.append(path_left + " " + path_right + " 0\n")
        dataset_idx = dataset_idx+1

        remain_num -= 1
        if dataset_idx% 1000 ==0:
            print(dataset_idx, "/", total_data_num)


with open('./ssimFlist', 'w') as f:
    f.writelines(new_flist)

    # for i in range(max_disp):
