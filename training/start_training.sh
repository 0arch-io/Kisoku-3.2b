#!/bin/bash
# Quick-start training script for 3.2B GPT on TPU v4-32

set -e

echo "=========================================="
echo "3.2B GPT TRAINING - TPU v4-32"
echo "=========================================="

# Check data is ready
echo "Checking dataset..."
WEB_COUNT=$(gsutil ls gs://openmind-2b-training-data/web/ | wc -l)
CODE_COUNT=$(gsutil ls -r gs://openmind-2b-training-data/code/ | grep tfrecord | wc -l)
MATH_COUNT=$(gsutil ls gs://openmind-2b-training-data/math/ | wc -l)
TOTAL=$((WEB_COUNT + CODE_COUNT + MATH_COUNT))

echo "Shards found: $TOTAL (web: $WEB_COUNT, code: $CODE_COUNT, math: $MATH_COUNT)"

if [ $TOTAL -lt 650 ]; then
  echo "⚠️  WARNING: Expected ~1000 shards for 50M sequences, found $TOTAL"
  echo "Data prep may still be running. Continue anyway? (y/n)"
  read -r response
  if [ "$response" != "y" ]; then
    echo "Aborting. Run this script again when data is ready."
    exit 1
  fi
fi

echo "✅ Dataset ready!"
echo ""

# Upload training script to all workers
echo "Uploading training script to TPU workers..."
gcloud compute tpus tpu-vm scp /tmp/train_3.2b.py openmind-2b:/tmp/ --zone us-central2-b --worker=all

echo ""
echo "Starting training on all 4 workers..."
echo "Logs will be in /tmp/training_*.log on each worker"
echo ""

# Start training on all workers
for w in 0 1 2 3; do
  echo "Starting worker $w..."
  gcloud compute tpus tpu-vm ssh openmind-2b \
    --zone us-central2-b \
    --worker=$w \
    --command "nohup python3 /tmp/train_3.2b.py > /tmp/training_w${w}.log 2>&1 &" &
done

wait

echo ""
echo "=========================================="
echo "✅ TRAINING STARTED ON ALL WORKERS"
echo "=========================================="
echo ""
echo "Monitor progress:"
echo "  gcloud compute tpus tpu-vm ssh openmind-2b --zone us-central2-b --worker=0 --command 'tail -f /tmp/training_w0.log'"
echo ""
echo "Check all workers:"
echo "  bash /tmp/check_training.sh"
echo ""
