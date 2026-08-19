#!/bin/bash
# Wrapper script to run GR00T memory calculator

cd /home/jz97/VLM_REPO/Isaac-GR00T

echo "=========================================="
echo "GR00T Memory Calculator"
echo "=========================================="
echo ""

# Use simplified version (no model loading required)
python tools/calculate_groot_memory_simple.py
