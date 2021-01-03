# -*- coding: utf-8 -*-
import requests
import os
from json import JSONDecoder
import time

http_url="https://api-cn.faceplusplus.com/facepp/v3/face/analyze"
key ="cahipPkAG0pMYvjd7OqamtcP05Ys41nR"
secret ="A46U31nvUZFS-MNatUf5krjvkBJW9si0"

def analyze_face(filepath,facetokens,i):
    data = {"api_key": key, "api_secret": secret, "face_tokens": facetokens,"return_attributes": "gender,age,smiling,headpose,facequality"}
    files = {"image_file": open(filepath, "rb")}
    response = requests.post(http_url, data=data, files=files)
    req_con = response.content.decode('utf-8')
    req_dict = JSONDecoder().decode(req_con)
    print(req_dict)
    print(req_dict['faces'][0]['attributes']['gender']['value'])
    print(req_dict['faces'][0]['attributes']['age']['value'])
    #print(req_dict['faces'][0]['gender'])
    with open("G:/f2fface.txt", 'a') as txt:
        for face in req_dict['faces']:
            #content = "{}".format(face)
            content="{},{}".format('m' if req_dict['faces'][0]['attributes']['gender']['value']=='Male' else 'f', req_dict['faces'][0]['attributes']['age']['value'])
            txt.write('%d'%i+','+content + '\n')

def detect_face(filepath):  # 传入图片文件
    http_url = "https://api-cn.faceplusplus.com/facepp/v3/detect"
    files = {"image_file": open(filepath, "rb")}
    data = {"api_key": key, "api_secret": secret}
    response = requests.post(http_url, data=data, files=files)
    req_dict = response.json()
    if req_dict['faces'] == []:
        return 'A'
    #print(req_dict)
    #print(req_dict['faces'][0]['face_token'])
    return req_dict['faces'][0]['face_token']
    # print(req_dict)

if __name__ == "__main__":
    path = "G:/result/f2f"
    #i=4
    # for i in range(1 , 5001):
    for i in range(2036, 5001):
        # img = '/'.join((path,'%d'%i,'face.jpg'))
        #img = '/'.join((path,'%d.jpg'%i))
        #img = '/'.join((path, 's2f%d.jpg' % i))
        img = '/'.join((path, 'f2f%d.jpg' % i))
        print(img)
        if os.path.exists(img) == False: continue
        print(i)

        a=detect_face(img)
        print('11111111111111')
        if a == 'A':
            with open("G:/f2fface.txt", 'a') as txt:
                content = "None"
                txt.write('%d' % i + ',' + content + '\n')
            continue
        analyze_face((img),a,i)
        time.sleep(1)

     #print(a)