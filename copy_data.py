import shutil
import os
from tqdm import tqdm

target_data_dir = '/data/01_experiment_data/img_snd/'
base_data_dir = 'data/'
all_data_txt = base_data_dir+'all_wav_16words.txt'
dest_data_dir = base_data_dir+'data_speech_commands_v0.02/'


def print_test(_list):
    print(_list)
    print(len(_list))

if __name__=='__main__':
    with open(all_data_txt) as f:
        lines = f.read().splitlines()
    
    for filename in tqdm(lines):
        dirname = os.path.dirname(filename)
        os.makedirs(dest_data_dir+dirname, exist_ok=True)
        shutil.copy(target_data_dir+filename, dest_data_dir+filename)
        spectogram_name = filename.replace('.wav', '.png')
        shutil.copy(target_data_dir+spectogram_name, dest_data_dir+spectogram_name)

    # print_test(lines)
    
    ### find entire class
    # class_list = [line.split('/')[0] for line in lines]
    # print_test(class_list)
    # print_test( list(set(class_list)) )
    # print(Counter(class_list))
