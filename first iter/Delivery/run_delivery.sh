#!/bin/bash

echo "======================================"
echo "  STARTING DELIVERY DRONE SYSTEM"
echo "======================================"

# Fail immediately if any command fails
set -e

# Go to script directory (important!)
cd "$(dirname "$0")"


echo "[INFO] Using Python:"
which python3

echo "[INFO] Starting delivery mission..."
python3 delivery_main.py

echo "[INFO] Delivery mission exited cleanly"
