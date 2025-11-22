#!/bin/bash
# Monitor the latest slurm output file

echo "Monitoring latest slurm output file..."
echo "Press Ctrl+C to stop"
echo ""

# Find the latest slurm file
LATEST_LOG=$(ls -t slurm-*.out 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "No slurm output files found yet. Waiting..."
    while [ -z "$LATEST_LOG" ]; do
        sleep 2
        LATEST_LOG=$(ls -t slurm-*.out 2>/dev/null | head -1)
    done
fi

echo "Found: $LATEST_LOG"
echo "================================"
tail -f "$LATEST_LOG"

