#!/bin/bash

# Define an array of your R script filenames
# Make sure to provide the correct path to your R scripts
R_SCRIPTS=(
    "best-subset-MW1.R"
    "best-subset-MW2.R"
    "best-subset-MW3.R"
    "best-subset-MW4.R"
    "best-subset-NE1.R"
    "best-subset-NE2.R"
    "best-subset-NE3.R"
    "best-subset-NE4.R"
    "best-subset-SE1.R"
    "best-subset-SE2.R"
    "best-subset-SE3.R"
    "best-subset-SE4.R"
    "best-subset-SW1.R"
    "best-subset-SW2.R"
    "best-subset-SW3.R"
    "best-subset-SW4.R"
    "best-subset-W1.R"
    "best-subset-W2.R"
    "best-subset-W3.R"
    "best-subset-W4.R"
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
