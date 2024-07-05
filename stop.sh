PS_NAME="poleapp.py"

PID=`ps -ef | grep ${PS_NAME} | grep -v 'grep' | awk '{print $2}'`
echo "Process ID: $PID"
if [[ "" !=  "$PID" ]]; then
	echo "Kill process"
	kill -9 $PID
else
	echo "No process is running"
fi
