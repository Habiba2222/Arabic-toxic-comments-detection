import tweepy
from tweepy import Stream
# from tweepy import api
# from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
import socket
import json


consumer_key = 'sq55SMdmlHaQV98YLDv1TabQc'
consumer_secret = 'vrDaja93Ml9jLBALT51cIuiiMEC0ZSMOp3VrKsc7PgirNzocY2'
access_token = '1535629326523179010-o8sww65HWyo9zGxRhSCgHUU9rpoqN6'
access_secret = 'FXRIfMn6awD9av7kZd60BNZxwPeIb4CGG2opREhDGBxNT'


class TweetsListener(Stream):

    def __init__(self, csocket):
        self.client_socket = csocket

    def on_data(self, data):
        try:
            msg = json.loads(data)
            print(msg['text'].encode('utf-8'))
            self.client_socket.send(msg['text'].encode('utf-8'))
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True


def sendData(username):
    userData = []
    # print("yes")
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
#   s=api.get_follower_ids(screen_name="GPtestingacc")
    user = api.get_user(screen_name=username)

    # print(user.statuses_count, user.followers_count, user.friends_count,
    #       user.favourites_count, user.listed_count, user.url, user.time_zone)

    data = []
    data.extend([user.statuses_count, user.followers_count, user.friends_count,
                 user.favourites_count, user.listed_count, user.url, user.time_zone])
    userData.append(data)
    return userData


if __name__ == "__main__":
    sendData()
