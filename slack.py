# -*- coding: utf-8 -*-
#변경금지
import requests
#메세지 전송하기---------------------------------------------------------------------------------------------
def slack(text):
    response = requests.post("https://slack.com/api/chat.postMessage",
        headers={"Authorization": "Bearer "+ "xoxb-2052427334483-4573998246807-hCIlOQIz4XpXNzAHzHEifu1I"},
        data={"channel":"#smartcs_pole","text":text})
        