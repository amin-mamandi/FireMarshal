#!/bin/bash

# Start attacker in background with large iteration count
./bankpll -m 40960 -a write -b 0x1e000 -l 16 -i 50000000 -c 0 -e 0 -x -s 1 -n 1 -A &
ATTACKER_PID=$!

# Run victim with small iteration count
./bankpll -m 40960 -a read -b 0x1e000 -l 16 -i 50 -c 3 -x -s 0 -n 1

# When victim finishes, terminate attacker and get its bandwidth
echo "Victim finished, terminating attacker..."
kill -TERM $ATTACKER_PID

# Wait for attacker to print its final bandwidth report
wait $ATTACKER_PID
echo "All processes completed"
m5 exit