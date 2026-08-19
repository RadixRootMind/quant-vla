# Environment Setup for GR00T + LIBERO

This directory contains environment configuration files for running GR00T inference and LIBERO evaluation.

## Quick Start (Recommended)

**One-click setup** - Run the automated setup script:

```bash
cd /home/jz97/VLM_REPO/Isaac-GR00T
bash setup_environments.sh
```

This will:
- ✅ Create `groot` conda environment
- ✅ Create `libero` conda environment
- ✅ Install all dependencies
- ✅ Configure LIBERO paths
- ✅ Verify installations

**Time**: ~10-15 minutes

---

## Manual Setup

If you prefer manual control or the script fails:

### Option 1: Using conda environment files

```bash
# Create groot environment
conda env create -f environments/groot_env.yml
conda activate groot
pip install -e .  # Install GR00T from source

# Create libero environment
conda env create -f environments/libero_env.yml
conda activate libero

# Install LIBERO
cd /tmp
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
pip install -e . --config-settings editable_mode=compat

# Configure LIBERO
mkdir -p ~/.libero
cat > ~/.libero/config.yaml <<EOF
assets: /tmp/LIBERO/libero/libero/assets
bddl_files: /tmp/LIBERO/libero/libero/bddl_files
benchmark_root: /tmp/LIBERO/libero/libero
datasets: /tmp/LIBERO/datasets
init_states: /tmp/LIBERO/libero/libero/init_files
EOF

mkdir -p /tmp/LIBERO/datasets
```

### Option 2: Using requirements.txt

```bash
# Create environments manually
conda create -n groot python=3.10 -y
conda create -n libero python=3.10 -y

# Install groot
conda activate groot
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
cd /home/jz97/VLM_REPO/Isaac-GR00T
pip install -e .

# Install libero
conda activate libero
pip install -r environments/requirements_libero.txt
# Then install LIBERO from source (see above)
```

---

## Files in This Directory

| File | Description |
|------|-------------|
| `groot_env.yml` | Conda environment file for GR00T inference server |
| `libero_env.yml` | Conda environment file for LIBERO evaluation |
| `requirements_libero.txt` | Pip requirements for LIBERO (alternative to yml) |
| `README.md` | This file |

---

## Verification

Test your environments:

```bash
# Test groot
conda activate groot
python -c "import torch; import transformers; import gr00t; print('✓ OK')"

# Test libero
conda activate libero
python -c "from libero.libero import get_libero_path; print('✓ OK')"
```

---

## Usage

### Terminal 1: Inference Server

```bash
conda activate groot
cd /home/jz97/VLM_REPO/Isaac-GR00T
./run_inference_server.sh libero_goal
```

### Terminal 2: Evaluation

```bash
conda activate libero
cd /home/jz97/VLM_REPO/Isaac-GR00T
./run_libero_eval.sh libero_goal --headless
```

---

## Troubleshooting

### Issue: `ModuleNotFoundError: transformers`

**Cause**: Running server in wrong environment
**Fix**: Make sure you're using `conda activate groot`

### Issue: `cannot import get_libero_path`

**Cause**: LIBERO not installed correctly
**Fix**:
```bash
conda activate libero
cd /tmp/LIBERO
pip install -e . --config-settings editable_mode=compat
```

### Issue: PyTorch version mismatch

**Cause**: Different CUDA version
**Fix**: Edit `groot_env.yml` and change:
- CUDA 11.8: `pytorch-cuda=11.8`
- CUDA 12.1: `pytorch-cuda=12.1`

### Issue: `WeightsUnpickler error` when loading LIBERO states

**Cause**: PyTorch 2.6 changed default `weights_only=True`
**Fix**: Already patched in setup script, or manually:
```python
# Edit /tmp/LIBERO/libero/libero/benchmark/__init__.py line 164
init_states = torch.load(init_states_path, weights_only=False)
```

---

## Updating Environments

To export your current environments for others:

```bash
# Export groot
conda activate groot
conda env export --no-builds > environments/groot_env_full.yml

# Export libero
conda activate libero
conda env export --no-builds > environments/libero_env_full.yml
```

---

## Clean Reinstall

If you need to start fresh:

```bash
conda remove -n groot --all -y
conda remove -n libero --all -y
rm -rf /tmp/LIBERO
rm -rf ~/.libero

# Then run setup again
bash setup_environments.sh
```

---

## System Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.10
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)
- **Disk Space**: ~15GB for both environments
- **RAM**: 16GB minimum, 32GB recommended

---

For detailed documentation, see `../SETUP_DUAL_ENV.md`
