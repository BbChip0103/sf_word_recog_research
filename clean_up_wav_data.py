import os
import os.path as path
import shutil
import librosa
from tqdm import tqdm

base_path = 'data'
data_path = path.join(base_path, 'data_speech_commands_v0.02')
strange_path = path.join(base_path, 'strange')

if __name__=='__main__':
    class_list = [filename for filename in os.listdir(data_path) 
                    if path.isdir(path.join(data_path, filename))]

    for _class in class_list:
        class_path = path.join(data_path, _class)
        wav_list = [filename for filename in os.listdir(class_path)
                                if filename.endswith('.wav')]
        for wav_filename in tqdm(wav_list):
            filename = path.join(class_path, wav_filename)
            wav, _ = librosa.load(filename, sr=16000)
            if wav.shape != (16000, ):
                strange_class_path = path.join(strange_path, _class)
                os.makedirs(strange_class_path, exist_ok=True)
                shutil.move(filename, strange_class_path)


    train_text_filename = path.join(base_path, 'train_16words.txt')
    validation_text_filename = path.join(base_path, 'validation_16words.txt')
    test_text_filename = path.join(base_path, 'test_16words.txt')
    with open(train_text_filename, 'r', encoding='utf-8') as f:
        train_16words_origin = f.read().splitlines()
    with open(validation_text_filename, 'r', encoding='utf-8') as f:
        validation_16words_origin = f.read().splitlines()
    with open(test_text_filename, 'r', encoding='utf-8') as f:
        test_16words_origin = f.read().splitlines()

    train_ok_text_filename = path.join(base_path, 'wav_train_16words_ok.txt')
    validation_ok_text_filename = path.join(base_path, 'wav_validation_16words_ok.txt')
    test_ok_text_filename = path.join(base_path, 'wav_test_16words_ok.txt')

    with open(train_ok_text_filename, 'w', encoding='utf-8') as f:
        for filename in tqdm(train_16words_origin):
            filename = filename.replace('/', path.sep)
            png_filename = path.join(data_path, filename)
            if path.isfile(png_filename):
                png_filename = png_filename.split(path.sep)[2:]
                png_filename = path.join(*png_filename)
                f.write(png_filename.replace(os.sep, '/')+'\n')

    with open(validation_ok_text_filename, 'w', encoding='utf-8') as f:
        for filename in tqdm(validation_16words_origin):
            filename = filename.replace('/', path.sep)
            png_filename = path.join(data_path, filename)
            if path.isfile(png_filename):
                png_filename = png_filename.split(path.sep)[2:]
                png_filename = path.join(*png_filename)
                f.write(png_filename.replace(os.sep, '/')+'\n')

    with open(test_ok_text_filename, 'w', encoding='utf-8') as f:
        for filename in tqdm(test_16words_origin):
            filename = filename.replace('/', path.sep)
            png_filename = path.join(data_path, filename)
            if path.isfile(png_filename):
                png_filename = png_filename.split(path.sep)[2:]
                png_filename = path.join(*png_filename)
                f.write(png_filename.replace(os.sep, '/')+'\n')