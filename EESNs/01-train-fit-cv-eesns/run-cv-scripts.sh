#!/bin/bash

# Define an array of your R script filenames
# Make sure to provide the correct path to your R scripts
R_SCRIPTS=(
    "eesns-MW-layers1.R"
    "eesns-MW-layers2.R"
    "eesns-MW-layers3.R"
    "eesns-NE-layers1.R"
    "eesns-NE-layers2.R"
    "eesns-NE-layers3.R"
    "eesns-SE-layers1.R"
    "eesns-SE-layers2.R"
    "eesns-SE-layers3.R"
    "eesns-SW-layers1.R"
    "eesns-SW-layers2.R"
    "eesns-SW-layers3.R"
    "eesns-W-layers1.R"
    "eesns-W-layers2.R"
    "eesns-W-layers3.R"
)

# Loop through each script and create a new tmux session
for SCRIPT in "${R_SCRIPTS[@]}"; do
    
    # Extract the script name without extension to use as the session name
    SESSION_NAME=$(basename "$SCRIPT" .R)
    
    # Check if a tmux session with the same name already exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Session $SESSION_NAME already exists. Skipping $SCRIPT"
    else
        # Create a new detached tmux session (-d) with a specific name (-s)
        # and execute the R script using Rscript
        tmux new-session -d -s "$SESSION_NAME" "Rscript $SCRIPT"
        echo "Started R script $SCRIPT in tmux session $SESSION_NAME"
    fi
done

echo "All R scripts are running in their respective tmux sessions."
echo "You can list sessions with 'tmux ls' and attach with 'tmux attach -t <session_name>'"
