#!/bin/bash
pkill -9 -f status_writer.py 2>/dev/null
sleep 1
nohup setsid /home/ec2-user/exp/.venv_connectome_app/bin/python /home/ec2-user/exp/pipeline/monitoring/status_writer.py >/tmp/status_writer.log 2>&1 </dev/null &
sleep 2
echo "status_writer launched, pid(s): $(pgrep -f status_writer.py | tr '\n' ' ')"
