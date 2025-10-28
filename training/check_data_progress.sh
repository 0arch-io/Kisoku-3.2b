#!/bin/bash
# Check data preparation progress

echo "=========================================="
echo "DATA PREPARATION PROGRESS"
echo "=========================================="
echo ""

# GCS stats
echo "Storage used:"
gsutil du -sh gs://openmind-2b-training-data/
echo ""

echo "Shard counts:"
WEB=$(gsutil ls gs://openmind-2b-training-data/web/ 2>/dev/null | wc -l)
CODE=$(gsutil ls -r gs://openmind-2b-training-data/code/ 2>/dev/null | grep tfrecord | wc -l)
MATH=$(gsutil ls gs://openmind-2b-training-data/math/ 2>/dev/null | wc -l)
TOTAL=$((WEB + CODE + MATH))

echo "  Web:  $WEB shards"
echo "  Code: $CODE shards"
echo "  Math: $MATH shards"
echo "  TOTAL: $TOTAL shards"
echo ""

# Expected: ~650 web shards (32.5M / 50K), ~250 code shards (12.5M / 50K), ~100 math shards (5M / 50K) = ~1000 total
EXPECTED=1000
PERCENT=$((TOTAL * 100 / EXPECTED))
echo "Progress: $PERCENT% of expected $EXPECTED shards"
echo ""

# Worker status
echo "=========================================="
echo "WORKER STATUS"
echo "=========================================="
for w in 0 1 2 3; do
  echo "Worker $w:"
  timeout 10 gcloud compute tpus tpu-vm ssh openmind-2b \
    --zone us-central2-b \
    --worker=$w \
    --command "ps aux | grep 'python3.*create_tfrecord\|python3.*process_extra' | grep -v grep | wc -l" 2>/dev/null || echo "  Unreachable"
done
echo ""

if [ $TOTAL -ge $EXPECTED ]; then
  echo "✅ DATA READY! Run: bash /tmp/start_training.sh"
else
  echo "⏳ Still preparing data... (~$((EXPECTED - TOTAL)) shards remaining)"
fi
