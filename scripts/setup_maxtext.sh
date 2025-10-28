#!/bin/bash
# Set up MaxText environment on TPU

cd ~

# Clone MaxText if needed
if [ ! -d "maxtext" ]; then
  git clone https://github.com/google/maxtext.git
fi

cd maxtext

# Create venv and install dependencies
python3 -m venv .venv
source .venv/bin/activate

# Install MaxText dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install datasets

echo "MaxText setup complete on $(hostname)"
