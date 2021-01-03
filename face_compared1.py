import face_recognition
import os


original_path = 'D:/BaiduNetdiskDownload/s2fTrueFace/s2fTrueFace'
target_path = 'E:/PycharmProjects/result/s2f1'
result = 0
i = 0
for filename1 in os.listdir(original_path):
    #print(filename1)
    filenamesplit = filename1.split('.')
    #print(filenamesplit)
    turepath = original_path + '/' + filename1
    print(turepath)
    generate_path = target_path+ '/' + 's2f' + filenamesplit[0]+'.jpg'
    print(generate_path)
    try:
        first_image = face_recognition.load_image_file(turepath)
        second_image = face_recognition.load_image_file(generate_path)
        first_encoding = face_recognition.face_encodings(first_image)[0]
        second_encoding = face_recognition.face_encodings(second_image)[0]
        results = face_recognition.compare_faces([first_encoding], second_encoding)
        print(results)
        if results == [True]:
            result += 1
        print(result)
    except:
        continue
print('result',result)



