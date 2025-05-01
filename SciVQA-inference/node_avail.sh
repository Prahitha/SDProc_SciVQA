#!/bin/bash
# for node in gpu{025..034}; do
#   echo -n "$node: "
#   scontrol show node $node | grep -i AllocTRES
# done


#!/bin/bash

# === Config ===
NODES=(gpu008 gpu009 gpu022 gpu023 gpu024 gpu025 gpu026 gpu027 gpu028 gpu029 gpu030 gpu031 gpu032 gpu033 gpu034 gpu035 gpu036 gpu037 gpu038 gpu039 gpu040 uri-gpu001 uri-gpu002 uri-gpu003 uri-gpu004 uri-gpu009)
MEM_THRESHOLD=60     # in GB
SLEEP_INTERVAL=60    # seconds between checks

echo "🔍 Watching GPU nodes (want ≥1 GPU & ≥${MEM_THRESHOLD}G RAM free)..."

while true; do
  echo -e "\n🕒 Checking at: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "NODE       GPU(F/T)   MEM_USED  MEM_FREE  STATUS"
  echo "--------------------------------------------------------"

  for node in "${NODES[@]}"; do
    info=$(scontrol show node "$node" 2>/dev/null)
    [[ -z "$info" ]] && continue

    total_gpu=$(echo "$info" | grep -oP "CfgTRES=.*?gres/gpu=\K[0-9]+" | head -n1)
    used_gpu=$(echo "$info" | grep -oP "gres/gpu=\K[0-9]+" | tail -n1)
    mem_used_mb=$(echo "$info" | grep -oP "mem=\K[0-9]+M" | head -n1 | tr -d M)
    mem_used_gb=$((mem_used_mb / 1024))
    mem_free_gb=$((515 - mem_used_gb))

    free_gpu=$((total_gpu - used_gpu))

    status="❌"
    if [[ $free_gpu -ge 1 && $mem_free_gb -ge $MEM_THRESHOLD ]]; then
      status="✅"
    fi

    printf "%-10s %d/%d        %4dG      %4dG    %s\n" "$node" "$free_gpu" "$total_gpu" "$mem_used_gb" "$mem_free_gb" "$status"
  done

  echo "--------------------------------------------------------"
  sleep "$SLEEP_INTERVAL"
done



# for node in gpu{008..009} gpu{022..024} gpu{025..034} gpu{056..057} arm-gpu{001..002} uri-arm-gpu001 uri-gpu{001..002} uri-gpu009; do
#   state=$(sinfo -n $node -o "%t" | tail -n 1)
#   if [[ "$state" == "idle" || "$state" == "mix" ]]; then
#     used=$(scontrol show node $node | grep -oP "GresUsed=gpu:\K[0-9]+" || echo 0)
#     total=$(scontrol show node $node | grep -oP "Gres=gpu:\K[0-9]+" || echo 0)
#     [[ -z $used ]] && used=0
#     [[ -z $total ]] && total=0
#     free=$((total - used))
#     if (( free > 0 )); then
#       echo "$node $free"
#     fi
#   fi
# done | sort -k2 -nr | head -n 5

