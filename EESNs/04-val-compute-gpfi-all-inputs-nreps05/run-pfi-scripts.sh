#!/bin/bash

# Define an array of your R script filenames
# Make sure to provide the correct path to your R scripts
R_SCRIPTS=(
    "pfi-MW1.R"
    "pfi-MW2.R"
    "pfi-MW3.R"
    "pfi-MW4.R"
    "pfi-NE1.R"
    "pfi-NE2.R"
    "pfi-NE3.R"
    "pfi-NE4.R"
    "pfi-SE1.R"
    "pfi-SE2.R"
    "pfi-SE3.R"
    "pfi-SE4.R"
    "pfi-SW1.R"
    "pfi-SW2.R"
    "pfi-SW3.R"
    "pfi-SW4.R"
    "pfi-W1.R"
    "pfi-W2.R"
    "pfi-W3.R"
    "pfi-W4.R"
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
