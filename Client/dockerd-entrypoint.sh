#!/bin/bash
sshpass -p 1 ssh -o "StrictHostKeyChecking no" root@10.103.0.3
tail -f /dev/null
