import numpy as np
import cv2
import os
import csv
import shutil
import scipy.io.wavfile as wavfile
import librosa
import random
import math
import matplotlib.pyplot as plt
import threading

def create_spectrogram(channel):
    complex_spectrogram = librosa.stft(channel, n_fft=512, hop_length=160, win_length=400, window='hamming')
    real, imag = complex_spectrogram.real[:,3:], complex_spectrogram.imag[:,3:]
    real = np.sign(real)* (np.abs(real)**0.3)
    imag = np.sign(imag) * (np.abs(imag) ** 0.3)
    real = np.expand_dims(real, axis=0)
    imag = np.expand_dims(imag, axis=0)
    spectrogram = np.concatenate((real,imag), axis=0)
    return spectrogram

def process(source, target, cx, cy):
    source_wavpath = '/'.join((source, 'speech.wav'))
    source_facepath = '/'.join((source, 'frameface.jpg'))
    frontalface_cascade = cv2.CascadeClassifier('C:/Users/hxs/Desktop/haarcascade_frontalface_default.xml')
    try:
        h, w, _ = (cv2.imread(source_facepath)).shape
    except AttributeError:
        return
    try:
        _, rawdata = wavfile.read(source_wavpath)
        rawdata = np.array(rawdata, dtype=float)
        duration = rawdata.shape[0]
        concat_n = math.ceil(96000 / duration)
    except ZeroDivisionError:
        return
    cx, cy = w*float(cx), h*float(cy)
    image_np = np.fromfile(source_facepath, dtype=np.uint8)
    image = cv2.imdecode(image_np, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = frontalface_cascade.detectMultiScale(gray, scaleFactor=1.10, flags=cv2.CASCADE_SCALE_IMAGE, minNeighbors=5, minSize=(80, 80))
    if len(faces) >= 1:
        for face in faces:
            if (face[0]<cx) and (cx<face[0]+face[2]) and (face[1]<cy) and (cy<face[1]+face[3]):
                x, y, w, h = face
                os.makedirs(target)
                face_cut = (cv2.imread(source_facepath))[y:y+h,x:x+w]
                resize = cv2.resize(src=face_cut, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite('/'.join((target, 'face.jpg')), resize)
                data= rawdata
                for i in range(concat_n - 1):
                    data = np.concatenate((data,rawdata),axis=0)
                # print(data.shape)
                if len(data.shape) == 1:
                    channel = data[0:96000]
                else:
                    channel = data[0:96000, 0]
                spectrogram = create_spectrogram(channel)
                print(spectrogram.shape)
                np.save('/'.join((target, 'speech6s.npy')), spectrogram)
                break

if __name__ == "__main__":
    idlist = {}
    nthread = 8
    with open("J:/avspeech_train.csv") as f:
        info = list(csv.reader(f))
    for i in info:
        time = i[1].split('.')[0]
        t = '_' + '0' * (3 - len(time)) + time
        name = i[0] + t
        idlist[name] = i[3:]
    path = "C:/Users/hxs/Desktop/avsp1"
    path2 = "C:/Users/hxs/Desktop/avsp_npyjpg1"
    if os.path.exists(path2) == False:
        os.makedirs(path2)

    for id in os.listdir(path):
        source = path+'/'+id
        target = path2+'/'+id
        print(id,"  start")
        if os.path.exists(target):
            print('ID exist, Skip!')
            continue
        if len(os.listdir(source)) < 2:
            print('file is missing!')
            continue
        cx, cy = idlist[id]
        while True:
            if len(threading.enumerate()) < nthread:
                break
        t = threading.Thread(target=process,args=(source, target, cx, cy))
        t.start()
