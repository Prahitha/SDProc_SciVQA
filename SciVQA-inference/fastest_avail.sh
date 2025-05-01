#!/bin/bash
squeue -t R -p gpu -o "%.18i %.8u %.9P %.20R %.10M %.10l %L %b" | \
grep -E "gpu0(09|10|2[2-9]|3[0-9]|04[0-1]|056|057|uri-gpu00[1-9])" | \
awk 'NR==1 {print; next} {
  split($7, a, "-");
  if (length(a) == 2) {
    days = a[1]; split(a[2], t, ":");
  } else {
    days = 0; split(a[1], t, ":");
  }
  total = days * 1440 + t[1] * 60 + t[2];
  print total, $0;
}' | sort -n | cut -d' ' -f2- | head -n 30
