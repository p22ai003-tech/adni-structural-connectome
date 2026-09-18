#!/bin/bash
PY=/home/ec2-user/exp/.venv_connectome_app/bin/python
while true; do printf '\033[3J\033[H\033[2J'; $PY /tmp/simple_mon.py 2>/dev/null; sleep 15; done
