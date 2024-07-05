#!/bin/bash
PS_NAME="poleapp.py"

PID=`ps -ef | grep ${PS_NAME} | grep -v 'grep' | awk '{print $2}'`
if [[ "" !=  "$PID" ]]; then
	echo "Already Running Process ID: $PID"
else
	echo "Starting ${PS_NAME}"
	nohup python3 ./${PS_NAME} &
fi
