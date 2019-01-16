from collections import Counter


base_data_dir = 'data/'
wav_data_dir = base_data_dir+'data_speech_commands_v0.02/'

def print_test(_list):
    print(_list)
    print(len(_list))

if __name__=='__main__':
    with open(base_data_dir+'validation_16words.txt') as f:
        lines = f.read().splitlines()
    # print_test(lines)
    
    ### find entire class
    class_list = [line.split('/')[0] for line in lines]
    # print_test(class_list)
    # print_test( list(set(class_list)) )
    print(Counter(class_list))
