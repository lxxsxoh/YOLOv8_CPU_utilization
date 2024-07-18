#!/bin/bash

# Check and create necessary directories if they don't exist
if [ ! -d "pid_results" ]; then
    mkdir pid_results
else
    rm -f pid_results/*.txt
fi

if [ ! -d "results" ]; then
    mkdir results
else
    rm -f results/*.txt
fi

if [ ! -d "mp_results" ]; then
    mkdir mp_results
else
    rm -f mp_results/*.txt
fi

# Function to run a script and monitor CPU usage with pidstat, mpstat
run_with_pidstat() {
    local script_name=$1
    local result_file="results/$2"
    local pidstat_file="pid_results/${script_name%.py}_pidstat.txt"
    local mpstat_file="mp_results/${script_name%.py}_mpstat.txt"
  
    # Start the python script and get its PID
    python3 $script_name > $result_file &
    SCRIPT_PID=$!

    # Monitor CPU usage of the script with pidstat including -I and -t options
    pidstat -h -r -u -I -t -p $SCRIPT_PID 2 > $pidstat_file &
    PIDSTAT_PID=$!

    # Monitor overall CPU usage with mpstat
    mpstat -P ALL 2 > $mpstat_file &
    MPSTAT_PID=$!

    # Wait for the python script to finish
    wait $SCRIPT_PID

    # Stop pidstat and mpstat monitoring
    kill $PIDSTAT_PID
    kill $MPSTAT_PID
}

# Run each script with pidstat, mpstat
run_with_pidstat "obb.py" "obb_result.txt"
run_with_pidstat "pose.py" "pose_result.txt"
run_with_pidstat "detection.py" "detect_result.txt"
run_with_pidstat "classification.py" "classify_result.txt"
run_with_pidstat "segment.py" "segment_result.txt"