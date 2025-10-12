#!/bin/bash
#
# DeepCompressor Auto-Install Script (No Conda)
# For PyTorch 2.8.0
#
# Usage: bash install_deepcompressor_no_conda.sh
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_DIR="/workspace"
PYTORCH_VERSION="2.8.0"
CUDA_VERSION="128"  # CUDA 12.8

print_header() {
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
}

print_step() {
    echo -e "${GREEN}[STEP $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Start installation
clear
print_header "DeepCompressor Installation (No Conda)"
echo ""
echo "This script will:"
echo "  1. Update system and install prerequisites"
echo "  2. Install Poetry"
echo "  3. Install PyTorch 2.8.0"
echo "  4. Clone DeepCompressor"
echo "  5. Fix dependencies"
echo "  6. Set exact package versions"
echo "  7. Install HuggingFace tools"
echo "  8. Install all packages"
echo "  9. Setup HuggingFace authentication"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# ============================================================================
# STEP 1: Update System and Install Prerequisites
# ============================================================================
print_step "1/13" "Updating system and installing prerequisites..."

apt-get update -qq
apt-get upgrade -y -qq

apt-get install -y -qq \
    git \
    wget \
    curl \
    build-essential \
    python3-dev \
    python3-pip \
    pkg-config \
    ffmpeg \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev

print_success "System packages installed"

# Verify Python
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_success "Python version: $PYTHON_VERSION"

# ============================================================================
# STEP 2: Install Poetry
# ============================================================================
print_step "2/13" "Installing Poetry..."

if command -v poetry &> /dev/null; then
    print_warning "Poetry already installed: $(poetry --version)"
else
    curl -sSL https://install.python-poetry.org | python3 - > /dev/null 2>&1
    export PATH="/root/.local/bin:$PATH"
    echo 'export PATH="/root/.local/bin:$PATH"' >> ~/.bashrc
    print_success "Poetry installed: $(poetry --version)"
fi

# ============================================================================
# STEP 3: Install PyTorch 2.8.0
# ============================================================================
print_step "3/13" "Installing PyTorch ${PYTORCH_VERSION}..."

# Check if PyTorch is already installed
if python3 -c "import torch" 2>/dev/null; then
    CURRENT_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null || echo "unknown")
    if [[ "$CURRENT_VERSION" == "$PYTORCH_VERSION"* ]]; then
        print_warning "PyTorch ${PYTORCH_VERSION} already installed"
    else
        print_warning "Current PyTorch: $CURRENT_VERSION, installing ${PYTORCH_VERSION}..."
        pip3 install --upgrade torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
    fi
else
    pip3 install torch==${PYTORCH_VERSION} torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}
fi

# Verify PyTorch installation
print_success "Verifying PyTorch installation..."
python3 << 'EOF'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
EOF

# ============================================================================
# STEP 4: Create Workspace and Clone Repository
# ============================================================================
print_step "4/13" "Setting up workspace and cloning repository..."

mkdir -p ${WORKSPACE_DIR}
cd ${WORKSPACE_DIR}

if [ -d "${WORKSPACE_DIR}/deepcompressor" ]; then
    print_warning "DeepCompressor already exists, pulling latest changes..."
    cd ${WORKSPACE_DIR}/deepcompressor
    git pull
else
    print_success "Cloning DeepCompressor..."
    git clone https://github.com/nunchaku-tech/deepcompressor.git
    cd ${WORKSPACE_DIR}/deepcompressor
fi

print_success "Repository ready at: $(pwd)"

# ============================================================================
# STEP 5: Fix pyproject.toml Bug
# ============================================================================
print_step "5/13" "Fixing pyproject.toml bug (pyav → av)..."

# Check if already fixed
if grep -q "^av = " pyproject.toml; then
    print_warning "pyproject.toml already fixed"
elif grep -q "^pyav = " pyproject.toml; then
    sed -i 's/pyav = ">= 13.0.0"/av = ">= 13.0.0"/' pyproject.toml
    print_success "Fixed: pyav → av"
else
    print_warning "Could not find pyav or av in pyproject.toml"
fi

# Verify the fix
if grep -q "^av = " pyproject.toml; then
    print_success "Verified: $(grep '^av = ' pyproject.toml)"
fi

# ============================================================================
# STEP 6: Set Exact Package Versions
# ============================================================================
print_step "6/13" "Setting exact package versions in pyproject.toml..."

# Set datasets version
if grep -q "^datasets = " pyproject.toml; then
    sed -i 's/^datasets = .*/datasets = "==3.6.0"/' pyproject.toml
    print_success "Set datasets==3.6.0"
fi

# Set diffusers version
if grep -q "^diffusers = " pyproject.toml; then
    sed -i 's/^diffusers = .*/diffusers = "==0.32.2"/' pyproject.toml
    print_success "Set diffusers==0.32.2"
fi

# Set transformers version
if grep -q "^transformers = " pyproject.toml; then
    sed -i 's/^transformers = .*/transformers = "==4.49.0"/' pyproject.toml
    print_success "Set transformers==4.49.0"
fi

print_success "Package versions configured"

# ============================================================================
# STEP 7: Install HuggingFace Tools
# ============================================================================
print_step "7/13" "Installing HuggingFace Hub and Transfer tools..."

pip3 install huggingface_hub hf_transfer --break-system-packages

# Verify installation
python3 -c "import huggingface_hub; print(f'  huggingface_hub version: {huggingface_hub.__version__}')"
print_success "HuggingFace tools installed successfully"

# ============================================================================
# STEP 8: Install PyAV
# ============================================================================
print_step "8/13" "Installing PyAV (av package)..."

pip3 install av>=13.0.0 --break-system-packages

# Verify
python3 -c "import av; print(f'  PyAV version: {av.__version__}')"
print_success "PyAV installed successfully"

# ============================================================================
# STEP 9: Configure Poetry
# ============================================================================
print_step "9/13" "Configuring Poetry..."

poetry config virtualenvs.in-project true
print_success "Poetry configured to use in-project virtualenvs"

# ============================================================================
# STEP 10: Install DeepCompressor Dependencies
# ============================================================================
print_step "10/13" "Installing DeepCompressor with Poetry (this may take several minutes)..."

cd ${WORKSPACE_DIR}/deepcompressor

# First, ensure poetry environment is created
poetry env use python3 2>/dev/null || true

# Pre-install av into poetry's venv to avoid wheel compatibility issues
print_success "Pre-installing av package into poetry environment..."
poetry run pip install av>=13.0.0

# Now install all other dependencies
poetry install

print_success "DeepCompressor installed"

# ============================================================================
# STEP 11: Verify Installation
# ============================================================================
print_step "11/13" "Verifying installation..."

poetry run python << 'EOF'
import torch
import av
import deepcompressor

errors = []

try:
    import transformers
    import diffusers
    import accelerate
    print("  ✓ All core packages imported successfully")
except ImportError as e:
    errors.append(str(e))

if torch.cuda.is_available():
    print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"  ✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    errors.append("CUDA not available")

if errors:
    print("\n  Errors found:")
    for error in errors:
        print(f"    - {error}")
    exit(1)
EOF

print_success "Installation verified"

# ============================================================================
# STEP 12: Set Environment Variables
# ============================================================================
print_step "12/13" "Setting environment variables..."

# Check if already set
if grep -q "PYTORCH_CUDA_ALLOC_CONF" ~/.bashrc; then
    print_warning "Environment variables already set in ~/.bashrc"
else
    cat >> ~/.bashrc << 'ENVEOF'

# PyTorch Memory Optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable TF32 for better A40 performance
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

# HuggingFace cache directory
export HF_HOME=/workspace/hf_cache

# HuggingFace transfer acceleration
export HF_HUB_ENABLE_HF_TRANSFER=1

ENVEOF
    print_success "Environment variables added to ~/.bashrc"
fi

# Apply now
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export HF_HOME=/workspace/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1

# Create HuggingFace cache directory
mkdir -p /workspace/hf_cache
print_success "Created HuggingFace cache directory"

# ============================================================================
# STEP 13: Create Test Script
# ============================================================================
print_step "13/13" "Creating test script..."

cat > ${WORKSPACE_DIR}/deepcompressor/test_installation.py << 'TESTEOF'
#!/usr/bin/env python3
"""Quick test of DeepCompressor setup"""

import torch
import deepcompressor
import os

def main():
    print("\n" + "=" * 60)
    print("DeepCompressor Installation Test")
    print("=" * 60)
    
    # Check GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Test tensor operations
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        z = torch.matmul(x, y)
        
        print(f"✓ GPU tensor operations working")
        print(f"  Allocated memory: {torch.cuda.memory_allocated(0) / 1e9:.3f} GB")
    
    # Check available examples
    examples_dir = "/workspace/deepcompressor/examples"
    if os.path.exists(examples_dir):
        print(f"\n✓ Examples directory found:")
        for item in os.listdir(examples_dir):
            print(f"  - {item}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
TESTEOF

chmod +x ${WORKSPACE_DIR}/deepcompressor/test_installation.py
print_success "Test script created"

# ============================================================================
# SETUP SHELL ENVIRONMENT
# ============================================================================

print_header "Finalizing Environment Setup"

# Source bashrc and ensure PATH is set
print_success "Applying environment changes..."
export PATH="/root/.local/bin:$PATH"

# Source bashrc in current shell
if [ -f ~/.bashrc ]; then
    source ~/.bashrc
    print_success "Environment loaded from ~/.bashrc"
fi

# Verify poetry is accessible
if command -v poetry &> /dev/null; then
    print_success "Poetry is now accessible: $(which poetry)"
else
    print_error "Poetry not found in PATH. You may need to restart your shell."
fi

# ============================================================================
# HUGGINGFACE AUTHENTICATION
# ============================================================================

print_header "HuggingFace Authentication"

echo ""
echo -e "${YELLOW}To use HuggingFace models, you need to authenticate.${NC}"
echo ""
echo "Options:"
echo "  1. Run now (interactive): hf auth login"
echo "  2. Run later with your token: huggingface-cli login"
echo ""
echo "Get your token from: https://huggingface.co/settings/tokens"
echo ""

read -p "Would you like to authenticate now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_success "Starting HuggingFace authentication..."
    huggingface-cli login
else
    print_warning "Skipping authentication. Run 'huggingface-cli login' when ready."
fi

# ============================================================================
# COMPLETION
# ============================================================================

print_header "Installation Complete!"

echo -e "${GREEN}✓ Installation successful!${NC}"
echo ""
echo "Quick Start Commands:"
echo "---------------------"
echo ""
echo "1. Run test script:"
echo "   cd /workspace/deepcompressor"
echo "   poetry run python test_installation.py"
echo ""
echo "2. Activate poetry environment:"
echo "   cd /workspace/deepcompressor"
echo "   poetry shell"
echo ""
echo "3. Check LLM examples:"
echo "   ls /workspace/deepcompressor/examples/llm/"
echo ""
echo "4. Check Diffusion examples:"
echo "   ls /workspace/deepcompressor/examples/diffusion/"
echo ""
echo "5. Monitor GPU:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "6. HuggingFace login (if not done):"
echo "   huggingface-cli login"
echo ""
echo "Documentation:"
echo "  - Setup guide: /workspace/deepcompressor/README.md"
echo "  - QServe paper: https://arxiv.org/abs/2405.04532"
echo "  - SVDQuant paper: https://arxiv.org/abs/2411.05007"
echo ""
echo "Environment variables set:"
echo "  - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo "  - HF_HOME=/workspace/hf_cache"
echo "  - HF_HUB_ENABLE_HF_TRANSFER=1"
echo ""
print_header "Happy Quantizing! 🚀"