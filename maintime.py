# -*- coding: utf-8 -*-
#변경금지
from pytz import timezone
from datetime import datetime
#현재시간 구하기-----------------------------------------------------------------------------
def maintime():
    KST = datetime.now(timezone('Asia/Seoul'))
    nowday = KST.strftime("%d")
    nowhour = KST.strftime("%H")
    nowminute = KST.strftime("%M")
    simplenowday = KST.strftime("%Y-%m-%d" )
    nowtime = KST.strftime("%Y-%m-%d %H:%M:%S")
    return nowday, nowhour, nowminute, simplenowday, nowtime

def sptime():
    KST = datetime.now(timezone('Asia/Seoul'))
    nowday = KST.strftime("%d")
    nowhour = KST.strftime("%H")
    nowminute = KST.strftime("%M")
    simplenowday = KST.strftime("%Y-%m-%d" )
    nowtime = KST.strftime("%Y-%m-%d %H:%M:%S")
    return nowhour, nowminute