from PIL import Image
import cv2
image_dir = r'E:\PycharmProjects\facedecoder\demo\1.jpg'
image = cv2.imread(image_dir)
print(image)