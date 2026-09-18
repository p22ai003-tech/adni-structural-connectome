#!/bin/bash
pkill -9 -f batch162_diskguard.py 2>/dev/null
nohup setsid /home/ec2-user/exp/.venv_connectome_app/bin/python /tmp/batch162_diskguard.py >/tmp/diskguard.log 2>&1 </dev/null &
