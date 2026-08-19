#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export AWESOME_QVLA_ROOT="${AWESOME_QVLA_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
export QUANTVLA_ROOT="${QUANTVLA_ROOT:-${AWESOME_QVLA_ROOT}}"

quantvla_find_conda_root() {
    local candidates=()

    if [[ -n "${CONDA_ROOT:-}" ]]; then
        candidates+=("${CONDA_ROOT}")
    fi
    if [[ -n "${CONDA_EXE:-}" ]]; then
        candidates+=("$(cd "$(dirname "${CONDA_EXE}")/.." && pwd)")
    fi
    candidates+=("${HOME}/miniconda3" "${HOME}/anaconda3" "/opt/conda")

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}/etc/profile.d/conda.sh" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "Unable to locate conda.sh. Set CONDA_ROOT explicitly." >&2
    return 1
}

quantvla_activate_env() {
    local env_name="$1"
    local conda_root
    conda_root="$(quantvla_find_conda_root)"
    # shellcheck disable=SC1090
    source "${conda_root}/etc/profile.d/conda.sh"
    conda activate "${env_name}"
    hash -r 2>/dev/null || true
}

quantvla_python_bin() {
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        if [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
            echo "${CONDA_PREFIX}/bin/python"
            return 0
        fi
        echo "Active conda environment is missing python: ${CONDA_PREFIX}/bin/python" >&2
        echo "Deactivate and recreate ${QUANTVLA_CONDA_ENV:-awesome_quant_vla}, or fix CONDA_ROOT/.env.local." >&2
        return 1
    fi

    local python_path
    python_path="$(command -v python || true)"
    if [[ -n "${python_path}" && -x "${python_path}" ]]; then
        echo "${python_path}"
        return 0
    fi

    echo "Unable to locate python for the active environment. Activate ${QUANTVLA_CONDA_ENV:-awesome_quant_vla} or set CONDA_ROOT correctly." >&2
    return 1
}

quantvla_default_conda_env() {
    echo "${AWESOME_QVLA_CONDA_ENV:-${QUANTVLA_CONDA_ENV:-awesome_quant_vla}}"
}

quantvla_default_libero_root() {
    local candidates=()

    if [[ -n "${LIBERO_ROOT:-}" ]]; then
        candidates+=("${LIBERO_ROOT}")
    fi
    candidates+=(
        "${QUANTVLA_ROOT}/../LIBERO"
        "${HOME}/LIBERO"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -d "${candidate}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done

    echo "${QUANTVLA_ROOT}/../LIBERO"
}

quantvla_export_pythonpath() {
    local libero_root
    local entries=("${QUANTVLA_ROOT}")

    libero_root="$(quantvla_default_libero_root)"
    export LIBERO_ROOT="${libero_root}"
    if [[ -d "${libero_root}" ]]; then
        entries+=("${libero_root}")
    fi

    if [[ -n "${PYTHONPATH:-}" ]]; then
        entries+=("${PYTHONPATH}")
    fi

    export PYTHONPATH
    PYTHONPATH="$(IFS=:; echo "${entries[*]}")"
}

quantvla_setup_cache_dirs() {
    local cache_root="${AWESOME_QVLA_CACHE_ROOT:-${QUANTVLA_CACHE_ROOT:-${HOME}/.cache/awesome_quant_vla}}"
    mkdir -p "${cache_root}/huggingface" "${cache_root}/torch" "${cache_root}/xdg"

    export AWESOME_QVLA_CACHE_ROOT="${cache_root}"
    export QUANTVLA_CACHE_ROOT="${cache_root}"
    export HF_HOME="${cache_root}/huggingface"
    export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
    export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
    export TORCH_HOME="${cache_root}/torch"
    export XDG_CACHE_HOME="${cache_root}/xdg"
}

quantvla_setup_libero_config() {
    local config_root="${LIBERO_CONFIG_PATH:-${HOME}/.libero}"
    local libero_root
    local benchmark_root
    local config_file
    mkdir -p "${config_root}"
    export LIBERO_CONFIG_PATH="${config_root}"

    libero_root="$(quantvla_default_libero_root)"
    benchmark_root="${libero_root}/libero/libero"
    config_file="${config_root}/config.yaml"

    if [[ -d "${benchmark_root}" ]]; then
        mkdir -p "${libero_root}/datasets"
        if [[ ! -f "${config_file}" || "${AWESOME_QVLA_REWRITE_LIBERO_CONFIG:-0}" = "1" ]]; then
            cat > "${config_file}" <<EOF
benchmark_root: ${benchmark_root}
bddl_files: ${benchmark_root}/bddl_files
init_states: ${benchmark_root}/init_files
datasets: ${libero_root}/datasets
assets: ${benchmark_root}/assets
EOF
        fi

        if [[ "${config_root}" != "${HOME}/.libero" ]]; then
            mkdir -p "${HOME}/.libero"
            if [[ ! -f "${HOME}/.libero/config.yaml" || "${AWESOME_QVLA_REWRITE_LIBERO_CONFIG:-0}" = "1" ]]; then
                cp "${config_file}" "${HOME}/.libero/config.yaml"
            fi
        fi
    fi
}
