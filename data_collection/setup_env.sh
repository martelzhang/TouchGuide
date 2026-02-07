#!/bin/bash

# Check the operating system
OS_NAME=$(uname -s)
OS_VERSION=""

if [[ "$OS_NAME" == "Linux" ]]; then
    if command -v lsb_release &>/dev/null; then
        OS_VERSION=$(lsb_release -rs)
    elif [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS_VERSION=$VERSION_ID
    fi
    if [[ "$OS_VERSION" != "22.04" ]]; then
        echo "Warning: This script has only been tested on Ubuntu 22.04"
        echo "Your system is running Ubuntu $OS_VERSION."
        read -p "Do you want to continue anyway? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Installation cancelled."
            exit 1
        fi
    fi
else
    echo "Unsupported operating system: $OS_NAME"
    exit 1
fi

echo "Operating system check passed: $OS_NAME $OS_VERSION"

# Resolve script directory so files can be referenced reliably
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Conda environment yaml file (canonical name)
CONDA_ENV_FILE="$SCRIPT_DIR/conda_environment.yaml"
if [[ ! -f "$CONDA_ENV_FILE" ]]; then
    echo "Error: conda environment yaml not found: $CONDA_ENV_FILE"
    exit 1
fi

# Function to create environment
create_environment() {
    local CONDA_CMD=$1
    local ENV_NAME=$2

    # Deactivate current environment if any (use conda deactivate for both conda and mamba)
    conda deactivate 2>/dev/null || true

    # Remove existing environment if it exists
    if $CONDA_CMD env list | grep -q "^$ENV_NAME "; then
        echo "Removing existing environment '$ENV_NAME'..."
        $CONDA_CMD env remove -n "$ENV_NAME" -y
    fi

    # Create new environment from conda_environment.yaml
    # If channel priority is strict, temporarily switch to flexible to allow mixing nvidia/conda-forge.
    local PREV_CHANNEL_PRIORITY=""
    local RESET_CHANNEL_PRIORITY=0
    PREV_CHANNEL_PRIORITY=$($CONDA_CMD config --show channel_priority 2>/dev/null | awk -F': ' '/channel_priority/{print $2}')
    if [[ "$PREV_CHANNEL_PRIORITY" == "strict" ]]; then
        echo "[INFO] Temporarily setting channel_priority to flexible for dependency solving..."
        $CONDA_CMD config --set channel_priority flexible
        RESET_CHANNEL_PRIORITY=1
    fi

    if ! $CONDA_CMD env create -f "$CONDA_ENV_FILE" -n "$ENV_NAME"; then
        if [[ $RESET_CHANNEL_PRIORITY -eq 1 ]]; then
            $CONDA_CMD config --set channel_priority "$PREV_CHANNEL_PRIORITY"
        fi
        echo "[ERROR] $CONDA_CMD env create failed."
        exit 1
    fi

    if [[ $RESET_CHANNEL_PRIORITY -eq 1 ]]; then
        $CONDA_CMD config --set channel_priority "$PREV_CHANNEL_PRIORITY"
    fi

    echo "$CONDA_CMD environment '$ENV_NAME' created from: $CONDA_ENV_FILE"

    echo -e "[INFO] Created $CONDA_CMD environment named '$ENV_NAME'.\n"
    echo -e "\t\t1. To activate the environment, run:                $CONDA_CMD activate $ENV_NAME"
    echo -e "\t\t2. To install conda dependencies, run:              bash $SCRIPT_NAME --install"
    echo -e "\t\t3. To deactivate the environment, run:              conda deactivate"
    echo -e "\n"
}

# Check if an environment name is provided
if [[ -n "$2" ]]; then
    ENV_NAME="$2"
else
    ENV_NAME="touch_guide_test"
fi

# Check if the --conda parameter is passed
if [[ "$1" == "--conda" ]]; then
    # Initialize conda
    if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        . "$HOME/anaconda3/etc/profile.d/conda.sh"
    else
        echo "Conda initialization script not found. Please install Miniconda3 or Anaconda3 or Miniforge3."
        exit 1
    fi
    create_environment "conda" "$ENV_NAME"

# Check if the --mamba parameter is passed
elif [[ "$1" == "--mamba" ]]; then
    # Initialize mamba (miniforge)
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        . "$HOME/miniforge3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/mambaforge/etc/profile.d/conda.sh" ]; then
        . "$HOME/mambaforge/etc/profile.d/conda.sh"
    else
        echo "Mamba initialization script not found. Please install Miniforge3 or Mambaforge."
        exit 1
    fi
    # Also source mamba.sh if available for full mamba support
    if [ -f "$HOME/miniforge3/etc/profile.d/mamba.sh" ]; then
        . "$HOME/miniforge3/etc/profile.d/mamba.sh"
    elif [ -f "$HOME/mambaforge/etc/profile.d/mamba.sh" ]; then
        . "$HOME/mambaforge/etc/profile.d/mamba.sh"
    fi
    create_environment "mamba" "$ENV_NAME"

# Check if the --install parameter is passed
elif [[ "$1" == "--install" ]]; then
    # Get the currently activated conda environment name
    if [[ -z "${CONDA_DEFAULT_ENV}" ]]; then
        echo "Error: No conda/mamba environment is currently activated."
        echo "Please activate an environment first with: conda/mamba activate <env_name>"
        exit 1
    fi
    ENV_NAME=${CONDA_DEFAULT_ENV}

    # Clean up any corrupted numpy installations in ~/.local first
    if [[ -d ~/.local/lib/python3.10/site-packages ]]; then
        CORRUPTED_PKGS=$(find ~/.local/lib/python3.10/site-packages -maxdepth 1 -name "-*" 2>/dev/null || true)
        if [[ -n "$CORRUPTED_PKGS" ]]; then
            echo "[INFO] Cleaning up corrupted packages in ~/.local/lib/python3.10/site-packages..."
            rm -rf ~/.local/lib/python3.10/site-packages/-* 2>/dev/null || true
            echo "[INFO] Corrupted packages cleaned up."
        fi
    fi

    # Detect conda/mamba command
    if [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
        CONDA_CMD="mamba"
    else
        CONDA_CMD="conda"
    fi

    echo "[INFO] Updating conda environment '$ENV_NAME' from: $CONDA_ENV_FILE"
    CONDA_CHANNEL_PRIORITY=flexible \
    $CONDA_CMD env update --repodata-fn repodata.json --repodata-fn repodata_from_packages.json \
        -f "$CONDA_ENV_FILE" -n "$ENV_NAME" || \
        echo "[WARN] Conda env update failed. Continuing with existing environment packages."

    # Install wxpython (needed for pysurvive/gooey/vive tracker) with full repodata
    WX_INSTALL_ENV="$CONDA_CMD install --repodata-fn repodata.json --repodata-fn repodata_from_packages.json -n \"$ENV_NAME\""
    if ! $CONDA_CMD list -n "$ENV_NAME" | grep -q "^wxpython "; then
        echo "[INFO] Installing wxpython from conda-forge (full repodata)..."
        if CONDA_CHANNEL_PRIORITY=flexible eval $WX_INSTALL_ENV -c conda-forge wxpython -y --freeze-installed; then
            echo "[INFO] wxpython installed successfully."
        else
            echo "[WARN] wxpython install failed; trying to update cuda-toolkit then retry..."
            if CONDA_CHANNEL_PRIORITY=flexible eval $WX_INSTALL_ENV -c nvidia "cuda-toolkit>=12.2,<12.3" -y --freeze-installed; then
                if CONDA_CHANNEL_PRIORITY=flexible eval $WX_INSTALL_ENV -c conda-forge wxpython -y --freeze-installed; then
                    echo "[INFO] wxpython installed successfully after cuda-toolkit bump."
                else
                    echo "[ERROR] wxpython installation via conda failed after retry (needed for Vive tracker)."
                    echo "        请检查网络或手动执行: CONDA_REPODATA_FNS=repodata.json,repodata_from_packages.json \\"
                    echo "          CONDA_CHANNEL_PRIORITY=flexible conda install -n $ENV_NAME -c conda-forge wxpython --freeze-installed"
                    exit 1
                fi
            else
                echo "[ERROR] 无法安装兼容的 cuda-toolkit (>=12.6.3,<12.7) 来继续安装 wxpython。"
                exit 1
            fi
        fi
    fi

    # Clean up corrupted numpy installation if present
    if pip list 2>&1 | grep -q "WARNING.*-umpy"; then
        echo "[INFO] Detected corrupted numpy installation, cleaning up..."
        pip uninstall -y numpy 2>/dev/null || true
        rm -rf ~/.local/lib/python3.10/site-packages/-umpy* 2>/dev/null || true
        echo "[INFO] Corrupted numpy cleaned up."
    fi

    pip install uv
    uv pip install --upgrade pip
    # Ensure editable installs (PEP 660) work: setuptools must provide build_editable
    uv pip install --upgrade "setuptools>=71.0.0,<81.0.0" wheel

    # Workaround for Python ctypes.util.find_library("udev") on conda envs:
    # Hacking the udev library discovery to avoid issues with pyudev/xensesdk.
    # If $CONDA_PREFIX/lib/udev exists as a directory, Python may return that directory as the "udev" library,
    # causing pyudev/xensesdk to crash with: "OSError: .../lib/udev: Is a directory".
    if [[ -n "${CONDA_PREFIX}" && -d "${CONDA_PREFIX}/lib/udev" ]]; then
        echo "[INFO] Fixing libudev discovery for pyudev (renaming ${CONDA_PREFIX}/lib/udev)..."
        if [[ -e "${CONDA_PREFIX}/lib/udev.rules.d" ]]; then
            mv "${CONDA_PREFIX}/lib/udev" "${CONDA_PREFIX}/lib/udev.rules.d.bak.$(date +%s)" || true
        else
            mv "${CONDA_PREFIX}/lib/udev" "${CONDA_PREFIX}/lib/udev.rules.d" || true
        fi
    fi
    if [[ -n "${CONDA_PREFIX}" && -e "${CONDA_PREFIX}/lib/libudev.so.1" && ! -e "${CONDA_PREFIX}/lib/libudev.so" ]]; then
        ln -s libudev.so.1 "${CONDA_PREFIX}/lib/libudev.so" || true
    fi

    # project root directory
    PROJECT_ROOT=$(pwd)
    ARX5_SDK_DIR="$PROJECT_ROOT/src/lerobot/robots/bi_arx5/ARX5_SDK"
    if [[ -d "$ARX5_SDK_DIR" ]]; then
        # Uninstall pip spdlog before building ARX5 SDK to avoid header conflicts
        # pip spdlog installs headers in $CONDA_PREFIX/include/python3.10/spdlog/ which conflicts
        # with conda's spdlog headers during ARX5 SDK compilation
        pip uninstall -y spdlog 2>/dev/null || true
        
        echo "[INFO] Building ARX5 SDK..."
        cd "$ARX5_SDK_DIR"
        rm -rf build || sudo rm -rf build
        mkdir -p build
        cd build
        cmake -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" ..
        make install -j4
        echo "[INFO] ARX5 SDK built successfully!"
        
        # Set real-time scheduling capability for Python (required by ARX5 SDK) if sudo works
        if sudo -n true 2>/dev/null; then
            echo "[INFO] Setting real-time scheduling capability for Python..."
            PYTHON_REAL_PATH=$(readlink -f "$CONDA_PREFIX/bin/python")
            sudo setcap cap_sys_nice=ep "$PYTHON_REAL_PATH"
            echo "[INFO] Real-time scheduling capability set for: $PYTHON_REAL_PATH"
        else
            echo "[WARN] sudo not available; skipping setcap cap_sys_nice for Python."
        fi
        
        # Create sitecustomize.py to preload conda's libstdc++ (fixes CXXABI version issues)
        echo "[INFO] Creating sitecustomize.py for C++ ABI compatibility..."
        PY_VER="$(python -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')"
        SITE_PACKAGES_DIR="${CONDA_PREFIX}/lib/${PY_VER}/site-packages"
        SITECUSTOMIZE_FILE="${SITE_PACKAGES_DIR}/sitecustomize.py"
        
        cat > "$SITECUSTOMIZE_FILE" << 'EOF'
"""
Sitecustomize for conda environment.

This file is automatically executed when Python starts.
It preloads the conda environment's libstdc++.so.6 to ensure C++ extensions
compiled with GCC 14.3.0 can find the required CXXABI_1.3.15 symbols.
"""
import os
import ctypes

conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    libstdcxx_path = os.path.join(conda_prefix, 'lib', 'libstdc++.so.6')
    if os.path.exists(libstdcxx_path):
        try:
            # Preload with RTLD_GLOBAL so all subsequently loaded modules can use it
            ctypes.CDLL(libstdcxx_path, mode=ctypes.RTLD_GLOBAL)
        except Exception:
            # Silently fail if preloading doesn't work
            pass
EOF
        echo "[INFO] sitecustomize.py created at: $SITECUSTOMIZE_FILE"
    fi
    cd "$PROJECT_ROOT"
    echo "[INFO] Installing Lerobot from pyproject.toml"

    # Use local UV cache to avoid permission issues
    export UV_CACHE_DIR="${PROJECT_ROOT}/.uvcache"
    mkdir -p "$UV_CACHE_DIR"

    if uv pip install -e .; then
        echo "[INFO] Lerobot installed successfully!"
    else
        echo "[ERROR] Lerobot installation failed. See the error output above."
        exit 1
    fi
    echo "[INFO] Installing xensesdk and xensegripper..."

    if uv pip install xensesdk xensegripper; then
        uv pip install av==15.1.0
        echo "[INFO] xensesdk and xensegripper installed successfully!"

        # Fix onnxruntime-gpu CUDA library loading issue
        # onnxruntime uses dlopen() to load CUDA provider, which doesn't respect LD_LIBRARY_PATH
        # We use patchelf to set RPATH so it can find CUDA libraries in conda environment
        echo "[INFO] Fixing onnxruntime-gpu RPATH for CUDA libraries..."
        uv pip install patchelf
        ONNX_CUDA_SO="${CONDA_PREFIX}/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime_providers_cuda.so"
        if [[ -f "$ONNX_CUDA_SO" ]]; then
            patchelf --set-rpath "${CONDA_PREFIX}/lib" "$ONNX_CUDA_SO"
            echo "[INFO] onnxruntime-gpu RPATH fixed: $(patchelf --print-rpath "$ONNX_CUDA_SO")"
        else
            echo "[WARN] onnxruntime CUDA provider not found, skipping RPATH fix."
        fi


        # Workaround:
        # After installing xensesdk, remove OpenCV's bundled Qt platform plugin if present.
        # This avoids Qt/XCB plugin loading issues inside conda environments.
        if [[ -n "${CONDA_PREFIX}" ]]; then
            PY_VER="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            QXCB_PATH="${CONDA_PREFIX}/lib/python${PY_VER}/site-packages/cv2/qt/plugins/platforms/libqxcb.so"
            QXCB_PATH_310="${CONDA_PREFIX}/lib/python3.10/site-packages/cv2/qt/plugins/platforms/libqxcb.so"

            if [[ -f "$QXCB_PATH" ]]; then
                echo "[INFO] Removing OpenCV Qt plugin: $QXCB_PATH"
                rm -f "$QXCB_PATH"
            elif [[ -f "$QXCB_PATH_310" ]]; then
                echo "[INFO] Removing OpenCV Qt plugin: $QXCB_PATH_310"
                rm -f "$QXCB_PATH_310"
            else
                echo "[INFO] OpenCV Qt plugin (libqxcb.so) not found; skipping removal."
            fi
        else
            echo "[WARN] CONDA_PREFIX is not set; cannot remove OpenCV Qt plugin."
        fi

    else
        echo "[ERROR] xensesdk/xensegripper installation failed. See the error output above."
        exit 1
    fi

    # Verify critical package versions
    echo "[INFO] Verifying package versions..."
    cd "$PROJECT_ROOT"
    TORCHCODEC_VER=$(python -c "import torchcodec; print(torchcodec.__version__)" 2>/dev/null || echo "NOT INSTALLED")
    AV_VER=$(python -c "import av; print(av.__version__)" 2>/dev/null || echo "NOT INSTALLED")
    TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT INSTALLED")
    echo "  - torch: $TORCH_VER"
    echo "  - torchcodec: $TORCHCODEC_VER (should be 0.7.0)"
    echo "  - av (pyav): $AV_VER (should be 15.1.0)"
    
    if [[ "$TORCHCODEC_VER" != "0.7.0" ]]; then
        echo "[WARN] torchcodec version mismatch! Expected 0.7.0, got $TORCHCODEC_VER"
        echo "[INFO] Attempting to fix torchcodec version..."
        uv pip install torchcodec==0.7.0 --force-reinstall
    fi
    
    if [[ "$AV_VER" != "15.1.0" ]]; then
        echo "[WARN] av (pyav) version mismatch! Expected 15.1.0, got $AV_VER"
        echo "[INFO] Attempting to fix av version..."
        uv pip install av==15.1.0 --force-reinstall
    fi
    
    echo "[INFO] ALL for TouchGuide installation completed successfully!"
    exit 0
else
    echo "Invalid argument. Usage:"
    echo "  --conda [env_name]   Create a conda environment (requires Miniconda/Anaconda)"
    echo "  --mamba [env_name]   Create a mamba environment (requires Miniforge)"
    echo "  --install            Install the package in the currently activated environment"
    exit 1
fi