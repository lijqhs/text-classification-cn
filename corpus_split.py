import os
import shutil
from const import category_labels

def split_corpus():
    # original data directory
    original_dataset_dir = './CN_Corpus/SogouC.reduced/Reduced'
    base_dir = 'data/'

    if (os.path.exists(base_dir)):
        print('`data` seems already exist.')
        return
    
    # make new folders
    os.mkdir(base_dir)
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

    # split corpus
    for cate in os.listdir(original_dataset_dir):
        cate_dir = os.path.join(original_dataset_dir, cate)
        file_list = os.listdir(cate_dir)
        print("cate: {}, len: {}".format(cate, len(file_list)))

        # train data
        fnames = file_list[:1500] 
        dst_dir = os.path.join(train_dir, category_labels[cate])
        os.mkdir(dst_dir)
        print("dst_dir: {}, len: {}".format(dst_dir, len(fnames)))
        for fname in fnames:
            src = os.path.join(cate_dir, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copyfile(src, dst)

        # test data
        fnames = file_list[1500:] 
        dst_dir = os.path.join(test_dir, category_labels[cate])
        os.mkdir(dst_dir)
        print("dst_dir: {}, len: {}".format(dst_dir, len(fnames)))
        for fname in fnames:
            src = os.path.join(cate_dir, fname)
            dst = os.path.join(dst_dir, fname)
            shutil.copyfile(src, dst)
    print('Corpus split DONE.')

split_corpus()