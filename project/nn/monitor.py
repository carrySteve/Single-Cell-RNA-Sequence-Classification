import os
import heapq

__FILES_TO_REMAIN = 5
__RESULT_PATH = '/root/steve/result/Graph read_pca-False use_pca-False n_components-20499 feature_dim-20499 embed_dim-2048/'


def file_name(file_dir):
    file_count = 0
    file_name_list = []
    acc_list = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            file_count += 1
            filename = os.path.splitext(file)
            if filename[1] == '.pth':
                acc = filename[0].split('-')[-1]
                acc_list.append(float(acc))
                file_name_list.append(file)
    
    if file_count > __FILES_TO_REMAIN:
        small_index_list = map(acc_list.index, heapq.nsmallest(file_count - __FILES_TO_REMAIN, acc_list))
        
        for idx in small_index_list:
            print(file_name_list[idx])
            os.remove(file_dir + file_name_list[idx])
    


if __name__ == "__main__":
    while(1):
        file_name(__RESULT_PATH)