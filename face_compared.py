import requests
import base64
import json
import os
# 1，准备好申请的人脸识别api，API Key， Secret Key
api1= 'https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials&client_id=udlS9NlMhLtvOHAN8ZXEGR0M&client_secret=lK7Sc2fTn1jMbM6oZi5rG5itty1rclPA'



# api2="https://aip.baidubce.com/rest/2.0/face/v3/match"

# 2,获取token值，拼接API
def get_token():
    response=requests.get(api1)
    access_token=eval(response.text)['access_token']
    api2="https://aip.baidubce.com/rest/2.0/face/v3/match"+"?access_token="+access_token
    return api2

# 3,读取图片数据
def read_img(img1,img2):
    with open(img1,'rb') as f:
        pic1=base64.b64encode(f.read())
    with open(img2,'rb') as f:
        pic2=base64.b64encode(f.read())
    params=json.dumps([
        {"image":str(pic1,"utf-8"),"image_type":'BASE64',"face_type":"LIVE"},
        {"image":str(pic2,"utf-8"),"image_type":'BASE64',"face_type":"IDCARD"}
    ])
    return params

# 4，发起请求拿到对比结果
def analyse_img(file1,file2):
    params=read_img(file1,file2)
    api=get_token()
    content=requests.post(api,params).text
    # print(content)
    score=eval(content)['result']['score']
    print(score)
    if score>50:
        print('图片识别相似度度为'+str(score)+',是同一人')
        return True
    else:
        print('图片识别相似度度为'+str(score)+',不是同一人')
        return False

res = analyse_img('D:/BaiduNetdiskDownload/s2fTrueFace/s2fTrueFace/1004.jpg','E:/PycharmProjects/result/s2f1/s2f1004.jpg')
# original_path = 'D:/BaiduNetdiskDownload/s2fTrueFace/s2fTrueFace'
# target_path = 'E:/PycharmProjects/result/s2f1'
# result = 0
# i = 0
# for filename1 in os.listdir(original_path):
#     #print(filename1)
#     filenamesplit = filename1.split('.')
#     #print(filenamesplit)
#     turepath = original_path + '/' + filename1
#     print(turepath)
#     generate_path = target_path+ '/' + 's2f' + filenamesplit[0]+'.jpg'
#     print(generate_path)
#     try:
#         i += 1
#         res = analyse_img(turepath,generate_path)
#         if res == True:
#             result += 1
#         print('-------',result)
#         print('-------', i)
#     except:
#         continue
# print('result',result)



