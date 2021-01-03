import os
import ffmpeg
import subprocess
import threading

path = "C:/Users/hxs/Desktop/avsp"
target = "C:/Users/hxs/Desktop/avsp1"

def extraction(video_path, wav_path, id):
    print(id)
    getwav = 'ffmpeg -i %s -f wav -ar 16000 -vn %s' % (video_path, wav_path)
    subprocess.call(getwav, shell=True)
    # getframe = 'ffmpeg -i %s -ss 00:00:00.1 -r 1 -an -q:v 2 ' % video_path + '/'.join((target, id, 'frame')) + '%03d.jpg'
    getframe = 'ffmpeg -i %s -f image2 ' % video_path + '/'.join((target, id, 'frameface.jpg'))
    subprocess.call(getframe, shell=True)

def begin(nthread=8):
    for video in os.listdir(path):
        wav_name = video[:-4]
        id = wav_name[:15]
        if os.path.exists(target+'/'+id) is not True:
            os.makedirs(target+'/'+id)
        video_path = path + '/' + video
        wav_path = '/'.join((target, id, 'speech.wav'))
        # extraction(video_path, wav_path, id)
        while True:
            if len(threading.enumerate()) < nthread:
                break
        t = threading.Thread(target=extraction,args=(video_path, wav_path, id))
        t.start()
if __name__ == "__main__":
    if os.path.exists(target) is not True:
        os.makedirs(target)
    begin()