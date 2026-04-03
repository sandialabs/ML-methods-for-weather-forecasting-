#!/bin/bash
# check_paths.sh

# 1. Setup the same paths as the Slurm script
PROJ_DIR="/home/mfholth/subseasonal/weekly_data/final_code"
export PYTHONPATH="${PROJ_DIR}/src:${PYTHONPATH}"

echo "--- Environmental Check ---"
echo "Project Directory: $PROJ_DIR"
echo "PYTHONPATH: $PYTHONPATH"

# 2. Test if Python can actually find your subseasonal package
echo -e "\n--- Testing Package Imports ---"
python -c "import subseasonal; print('✅ subseasonal package found at:', subseasonal.__file__)" || echo "❌ subseasonal package NOT found"
python -c "import subseasonal; import os; print('Location:', subseasonal.__path__ if hasattr(subseasonal, '__path__') else 'No Path'); print('File:', getattr(subseasonal, '__file__', 'None'))"