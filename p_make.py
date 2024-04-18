import glob
import os
import shutil

category_txt = 'project/Boundless-in-Pytorch-master/data_256_category.txt'
data256_folder = 'project/Boundless-in-Pytorch-master/data_256'
output_folder = 'project/Boundless-in-Pytorch-master/img_train'
img_per = 201 #obtain X img from each category in download_list.txt

with open(category_txt) as f:
    w = f.read()
    w = w.split('\n')
    n = 0
    print(w)
    for i in w:
        if os.path.exists(data256_folder + '/' + i[0] + '/' + i):
            img_list = sorted(list(glob.glob(data256_folder + '/' + i[0] + '/' + i + "/*.*")))
            for j in range(img_per): 
                shutil.copy(img_list[j], output_folder + "/" + str(n) + '.png')
                n = n + 1
        else:
            print(data256_folder + '/' + i[0] + '/' + i + ' does not exist')
