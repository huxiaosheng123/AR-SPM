# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# # @Time    : 2018/1/2 18:43
# # @Author  : He Hangjiang
# # @Site    :
# # @File    : CreatFcaeSet.py
# # @Software: PyCharm
#
# import requests
# from json import JSONDecoder
# import cv2
#
# http_url = "https://api-cn.faceplusplus.com/facepp/v3/faceset/create"
# key = "2PX56nNhKpVd57QNKc9sQt9ASRczWlIr"
# secret = "7gwhn-gVp6J3we_vdj_bIlL1bGpuZdlp"
#
# data = {"api_key": key, "api_secret": secret, 'display_name':'FacesStore',"outer_id":"hhj"}
#
# response = requests.post(http_url, data=data)
# print(response)
# print(response.text)
#
# # req_con = response.content.decode('utf-8')
# # req_dict = JSONDecoder().decode(req_con)
# #
# # print(req_dict)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/2 19:12
# @Author  : He Hangjiang
# @Site    :
# @File    : SearchFace.py
# @Software: PyCharm

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/2 18:55
# @Author  : He Hangjiang
# @Site    :
# @File    : AddFace.py
# @Software: PyCharm

# import requests
# from json import JSONDecoder
# import cv2
#
# http_url = "https://api-cn.faceplusplus.com/facepp/v3/faceset/addface"
# key = "2PX56nNhKpVd57QNKc9sQt9ASRczWlIr"
# secret = "7gwhn-gVp6J3we_vdj_bIlL1bGpuZdlp"
#
# data = {"api_key": key, "api_secret": secret, "outer_id": "hhj","face_tokens":'d6eee462bc73858ae8ed7accca1f6632,c06b924cb227711477291c346b2cb40f'}
#
# # filepath = "face2.jpg"
#
# # files = {"image_file": open(filepath, "rb")}
# response = requests.post(http_url, data=data)
# print(response)
# print(response.text)
#
# # req_con = response.content.decode('utf-8')
# # req_dict = JSONDecoder().decode(req_con)
# #
# # print(req_dict)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/1/2 19:12
# @Author  : He Hangjiang
# @Site    :
# @File    : SearchFace.py
# @Software: PyCharm

import requests
from json import JSONDecoder
import cv2

http_url = "https://api-cn.faceplusplus.com/facepp/v3/search"
key = "2PX56nNhKpVd57QNKc9sQt9ASRczWlIr"
secret = "7gwhn-gVp6J3we_vdj_bIlL1bGpuZdlp"
filepath = "E:/PycharmProjects/facedecoder/demo/1.jpg"

data = {"api_key": key, "api_secret": secret, "outer_id": "hhj"}
files = {"image_file": open(filepath, "rb")}
response = requests.post(http_url, data=data, files=files)

req_con = response.content.decode('utf-8')
req_dict = JSONDecoder().decode(req_con)
print(req_dict)

faces_token = ["d6eee462bc73858ae8ed7accca1f6632","5e85caf4c983b9523f17bebd77715d17"]

if req_dict["results"][0]["face_token"] in faces_token and req_dict["results"][0]["confidence"]>=req_dict["thresholds"]["1e-5"]:
    print("是番茄")
else:
    print("不是番茄")