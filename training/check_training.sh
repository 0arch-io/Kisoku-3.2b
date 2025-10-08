#!/bin/bash
# Check training progress across all workers

echo "=========================================="
echo "TRAINING STATUS CHECK"
echo "=========================================="
echo ""

for w in 0 1 2 3; do
  echo "=== WORKER $w ==="
  timeout 15 gcloud compute tpus tpu-vm ssh openmind-2b \
    --zone us-central2-b \
    --worker=$w \
    --command "echo 'Process:' && ps aux | grep 'python3.*train_3.2b.py' | grep -v grep || echo 'Not running' && echo '' && echo 'Latest logs:' && tail -20 /tmp/training_w${w}.log 2>/dev/null || echo 'No logs yet'" 2>/dev/null || echo "Worker $w unreachable"
  echo ""
done

echo "=========================================="
echo "GCS Checkpoints:"
gsutil ls gs://openmind-2b-training-data/ 2>/dev/null | grep -i checkpoint || echo "No checkpoints yet"
echo "=========================================="
