#!/bin/bash
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-MW1.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-MW2.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-MW3.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-MW4.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-NE1.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-NE2.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-NE3.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-NE4.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-SE1.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-SE2.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-SE3.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-SE4.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-SW1.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-SW2.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-SW3.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-SW4.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-W1.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-W2.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-W3.R &
nohup R CMD BATCH --no-save --no-restore 02.2-BestSubset-W4.R &

echo "All R scripts have been submitted."
