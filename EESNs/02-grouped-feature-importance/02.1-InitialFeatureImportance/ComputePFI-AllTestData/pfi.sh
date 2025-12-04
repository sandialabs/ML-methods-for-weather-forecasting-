#!/bin/bash
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-MW1.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-MW2.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-MW3.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-MW4.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-NE1.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-NE2.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-NE3.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-NE4.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-SE1.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-SE2.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-SE3.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-SE4.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-SW1.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-SW2.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-SW3.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-SW4.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-W1.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-W2.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-W3.R &
nohup R CMD BATCH --no-save --no-restore 02.1-ComputePFI-AllTestData-W4.R &
echo "All R scripts have been submitted."