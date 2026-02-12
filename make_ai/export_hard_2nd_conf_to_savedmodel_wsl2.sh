#!/bin/bash
# Run SavedModel export inside WSL2 (same TF/Keras runtime used for training).

set -euo pipefail

cd /mnt/c/Users/slhg1/OneDrive/Desktop/SMARTCS_Pole

if [ -d "venv_wsl2" ]; then
  # shellcheck disable=SC1091
  source venv_wsl2/bin/activate
else
  echo "venv_wsl2 not found. Run setup_wsl2_venv.sh first."
  exit 1
fi

echo "Export hard-2nd conf models to SavedModel..."
python3 make_ai/export_hard_2nd_conf_to_savedmodel.py --local "$@"
