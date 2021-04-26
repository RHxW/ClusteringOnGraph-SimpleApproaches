import os
import shutil
import datetime

def get_results():
    list_file = ""
    meta_file = ""
    res_dir = ""

    if not os.path.exists(res_dir):
        os.mkdir(res_dir)

    with open(list_file, "r") as f:
        list_lines = f.readlines()

    with open(meta_file, "r") as f:
        meta_lines = f.readlines()

    while list_lines[-1] == "":
        list_lines = list_lines[:-1]
    while meta_lines[-1] == "":
        meta_lines = meta_lines[:-1]

    label_dirs = []
    labels = set()

    for i in range(len(meta_lines)):
        label = meta_lines[i]
        if label[-1] == "\n":
            label = label[:-1]
        if not label:
            continue
        save_dir = res_dir + label + "/"
        if label not in labels:
            labels.add(label)
            os.mkdir(save_dir)
        # if label not in label_dirs:
        #     os.mkdir(save_dir)
        #     label_dirs.append(label)
        img_path = list_lines[i]
        if img_path[-1] == "\n":
            img_path = img_path[:-1]
        img_name = img_path.split("/")[-1]
        shutil.copy(img_path, save_dir + img_name)

    get_singular(res_dir)

def get_singular(res_root):
    if not os.path.exists(res_root):
        return
    if res_root[-1] != "/":
        res_root += "/"

    id_dirs = os.listdir(res_root)
    singular_dir = res_root + "Singular/"
    if not os.path.exists(singular_dir):
        os.mkdir(singular_dir)
    for id_dir in id_dirs:
        id_path = res_root + id_dir + "/"
        imgs = os.listdir(id_path)
        if len(imgs) < 2:
            for img in imgs:
                shutil.move(id_path + img, singular_dir + img)
            shutil.rmtree(id_path)

if __name__ == "__main__":
    get_results()