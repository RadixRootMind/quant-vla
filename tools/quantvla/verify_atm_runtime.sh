#!/bin/bash
# Script to verify ATM is properly loaded during GR00T inference
# Run this AFTER starting the inference server

echo "=========================================="
echo "GR00T ATM Runtime Verification"
echo "=========================================="

# Check environment variables
echo ""
echo "1. Environment Variables:"
echo "  GR00T_ATM_ENABLE: ${GR00T_ATM_ENABLE:-NOT SET}"
echo "  GR00T_ATM_ALPHA_PATH: ${GR00T_ATM_ALPHA_PATH:-NOT SET}"
echo "  GR00T_ATM_SCOPE: ${GR00T_ATM_SCOPE:-dit (default)}"

if [[ "${GR00T_ATM_ENABLE}" != "1" ]]; then
    echo ""
    echo "❌ WARNING: GR00T_ATM_ENABLE is not set to 1!"
    echo "   ATM will NOT be loaded."
    echo "   Fix: export GR00T_ATM_ENABLE=1"
fi

if [[ -z "${GR00T_ATM_ALPHA_PATH}" ]]; then
    echo ""
    echo "❌ WARNING: GR00T_ATM_ALPHA_PATH is not set!"
    echo "   ATM will NOT be loaded."
    echo "   Fix: export GR00T_ATM_ALPHA_PATH=/path/to/atm_alpha.json"
elif [[ ! -f "${GR00T_ATM_ALPHA_PATH}" ]]; then
    echo ""
    echo "❌ WARNING: Alpha JSON file not found!"
    echo "   Path: ${GR00T_ATM_ALPHA_PATH}"
fi

# Check if inference server is running
echo ""
echo "2. Checking Running Processes:"
inference_pid=$(ps aux | grep "inference_service.py" | grep -v grep | awk '{print $2}' | head -1)

if [[ -z "$inference_pid" ]]; then
    echo "  ⚠️  No inference_service.py process found"
    echo "     Cannot check runtime logs"
else
    echo "  ✅ Found inference server (PID: $inference_pid)"

    # Try to find log file (common locations)
    log_candidates=(
        "/tmp/groot_inference_${inference_pid}.log"
        "~/groot_inference.log"
        "groot_inference.log"
    )

    log_file=""
    for candidate in "${log_candidates[@]}"; do
        if [[ -f "$candidate" ]]; then
            log_file="$candidate"
            break
        fi
    done

    if [[ -n "$log_file" ]]; then
        echo ""
        echo "3. Checking Runtime Logs ($log_file):"

        # Check for ATM loading messages
        if grep -q "GR00T-ATM.*ATM enabled" "$log_file"; then
            echo "  ✅ ATM was loaded!"
            grep "GR00T-ATM.*ATM enabled" "$log_file" | tail -1
        else
            echo "  ❌ No ATM loading message found in logs"
            echo "     Expected to see: [GR00T-ATM] ATM enabled for X layers"
        fi

        # Check for ATM warnings
        if grep -q "GR00T-ATM.*warning\|GR00T-ATM.*skipped" "$log_file"; then
            echo ""
            echo "  ⚠️  ATM Warnings/Errors:"
            grep "GR00T-ATM" "$log_file" | grep -i "warning\|error\|skip" | tail -5
        fi
    else
        echo ""
        echo "3. Log File:"
        echo "  ⚠️  Could not find log file automatically"
        echo "     Please check your inference server output manually"
        echo "     Look for: [GR00T-ATM] ATM enabled for X layers"
    fi
fi

# Provide manual check instructions
echo ""
echo "=========================================="
echo "Manual Verification Steps:"
echo "=========================================="
echo "1. When starting inference server, check the output for:"
echo "   [GR00T-ATM] ATM enabled for 16 layers (512 heads) using ..."
echo ""
echo "2. If you see: [GR00T-ATM] No attention layers matched alpha JSON"
echo "   -> Layer name mismatch problem"
echo ""
echo "3. If you see: [GR00T-ATM] ATM enable requested but ... not set"
echo "   -> Environment variable issue"
echo ""
echo "4. If you don't see ANY [GR00T-ATM] messages:"
echo "   -> ATM is not being initialized at all"
echo "   -> Check GR00T_ATM_ENABLE=1 is set BEFORE starting server"
echo "=========================================="
