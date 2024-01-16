from matplotlib.font_manager import json_dump
from pydantic import Json
import requests
import speech_recognition as sr
from moviepy.editor import AudioFileClip
from flask import current_app
from app import *
from machine_model.Fake import detectAccount
from tweetAPI import sendData
from machine_model.Classification import testing
video = Blueprint('video', __name__)
video_url_default = "https://cdn.syndication.twimg.com/tweet?id="

###################################################################################################################
#####################################################################################################################


@video.route('/video/test', methods=["GET"])
def test():
    return jsonify({"message": "hello to video route"})

###################################################################################################################
#####################################################################################################################


@video.route('/prediction/txt', methods=["post"])
def predict():

    output = []
    # print(request.get_json())
    videopred = ""
    tweetpred = ""
    tweet = request.get_json()["predict"]
    videoId = request.get_json()["video_id"]
    isVideo = request.get_json()["is_video"]
    print("TWEET", tweet)
    print("VIDEOID", videoId)
    print("ISVIDEO", isVideo)
  #  for i in range(len(predict)):

 ######################################################################################################
    if(isVideo):
        transcribed_audio_file_name = videoId + ".wav"
        # print(request.get_json()["url"])
        video_url = requests.get(video_url_default+videoId)
        # "https://video.twimg.com/amplify_video/1516098066364907527/vid/1280x720/AVj-FHcdOqwwYL0_.mp4?tag=14"
        video_url = video_url.json()
        audioclip = AudioFileClip(video_url["video"]["variants"][1]["src"])
        audioclip.write_audiofile(transcribed_audio_file_name)

        r = sr.Recognizer()
        audio_ex = sr.AudioFile(transcribed_audio_file_name)
        type(audio_ex)

        with audio_ex as source:
            audiodata = r.record(source, duration=15)
        type(audiodata)

        # Extract text
        text = r.recognize_google(audio_data=audiodata, language='ar-AR')
        print("VIDEO TEXT", text)

        # f = open("transcription.txt", "a", encoding="utf-8")
        # f.write(text)
        if (text == ""):
            videopred = "NOT_HS"
            print("video pred1", videopred)
        else:
            videopred = testing(text)
            print("video pred2", videopred)

    if(tweet):
        tweetpred = testing(tweet)
        print("tweet pred1", tweetpred)
    else:
        tweetpred = "NOT_HS"
        print("tweet pred2", tweetpred)

    return jsonify({"videoOutput": videopred, "tweetOutput": tweetpred})
###########################################################################################################
    # output.append(pred)
    # return jsonify({"output": pred})
#####################################################################################

######################################################################################################
##########################################################################################################


@video.route('/detect/fake', methods=["post"])
def detectFake():

    print(request.get_json())
    username = request.get_json()["username"]
    print(username)
    userData = sendData(username)
    pred = detectAccount(userData)
    return jsonify({"fake": pred})


# pred = detectAccount([[21, 0, 7, 0, 0, None, None]])
# v = ""
# if pred == "1":
#     v = "Fake"
# print("The prediction", v)
# ######################################################################################################
##########################################################################################################
# @video.route('/video/video_text', methods=["POST"])
# def video_text():
#     # request.get_json()["url"]=====>ID
#     transcribed_audio_file_name = request.get_json()["url"] + ".wav"
#     print(request.get_json()["url"])
#     video_url = requests.get(video_url_default+request.get_json()["url"])
#     # "https://video.twimg.com/amplify_video/1516098066364907527/vid/1280x720/AVj-FHcdOqwwYL0_.mp4?tag=14"
#     video_url = video_url.json()
#     audioclip = AudioFileClip(video_url["video"]["variants"][1]["src"])
#     audioclip.write_audiofile(transcribed_audio_file_name)

#     r = sr.Recognizer()
#     audio_ex = sr.AudioFile(transcribed_audio_file_name)
#     type(audio_ex)

#     with audio_ex as source:
#         audiodata = r.record(source, duration=15)
#     type(audiodata)

#     # Extract text
#     text = r.recognize_google(audio_data=audiodata, language='ar-AR')

#     # f = open("transcription.txt", "a", encoding="utf-8")
#     # f.write(text)
#     if (text ==""):
#         pred = testing(text)
#     else:
#         pred="NOT_HS"


#     # f.close()

#     return jsonify({"output": pred})


# resp=video_text("1486125709202362371")
# print("responsse",resp)
