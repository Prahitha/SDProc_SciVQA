#!/bin/bash
# Get the squeue output and process it
squeue -t R -p gpu -o "%.18i %.8u %.9P %.20R %.10M %.10l %L %b" | \
grep -E "gpu0(09|10|2[2-9]|3[0-9]|04[0-1]|056|057|uri-gpu00[1-9])" | \
awk 'NR==1 {print; next} {
  # Store the original line
  line = $0
  
  # Parse the remaining time (column 7) to convert to minutes for consistent sorting
  remaining = $7
  rmins = 0
  
  # Handle days format: D-HH:MM:SS
  if (match(remaining, /([0-9]+)-([0-9]+):([0-9]+):([0-9]+)/, arr)) {
    rmins = (arr[1] * 24 * 60) + (arr[2] * 60) + arr[3]
  }
  # Handle hours format: HH:MM:SS
  else if (match(remaining, /([0-9]+):([0-9]+):([0-9]+)/, arr)) {
    rmins = (arr[1] * 60) + arr[2]
  }
  
  # Output with remaining time in minutes as a sortable prefix, then the original line
  printf "%012d\t%s\n", rmins, line
}' | \
sort -n | \
cut -f2- | \
head -n 30