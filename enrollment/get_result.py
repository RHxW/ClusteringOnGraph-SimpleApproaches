import os
import shutil

def get_result():
    meta_file = "F:/FaceClusterEnrollment/src/test_res/weekly_facepic_res/20210325140221.txt"
    with open(meta_file, "r") as f:
        meta_data = f.readlines()

    origin_root = "G:/facepic_weekly/20210310/face_all_align/"
    dst_root = "F:/FaceClusterEnrollment/src/test_res/weekly_facepic_res/20210325140221/"
    
    if origin_root[-1] != "/":
        origin_root += "/"
    if dst_root[-1] != "/":
        dst_root += "/"

    N = len(meta_data)
    for i in range(N):
        if meta_data[-1] == "\n":
            meta_data = meta_data[:-1]
            new_name = metadata.split(" ")[0]
            o_name = new_name.split("_")[-1].replace("$", "_")
            new_id = meta_data.split(" ")[-1]
            new_dir = dst_root + new_id + "/"
            if not os.path.exists(new_dir):
                os.mkdir(new_dir)
            shutil.copy(origin_root + o_name, new_dir + new_name)

if __name__ == "__main__":
    get_result()