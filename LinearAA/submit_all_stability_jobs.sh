#!/bin/bash

# Loop for specific k values
for k in 13 15 17 19 21 25 27
do
    echo "Submitting job for k=$k..."
    
    # Create a temporary job script for this k
    cat <<EOF > stab_job_k${k}.bsub
#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J Stability_k${k}
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- specify that the cores must be on the same host --
#BSUB -R "span[hosts=1]"
### -- specify that we need 4GB of memory per core/slot --
#BSUB -R "rusage[mem=4GB]"
### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot --
#BSUB -M 5GB
### -- set walltime limit: hh:mm --
#BSUB -W 24:00
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s243225@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o output_logs/Stability_k${k}_%J.out
#BSUB -e output_logs/Stability_k${k}_%J.err

# Create output directory for logs
mkdir -p output_logs

# GPU Request
#BSUB -gpu "num=1:mode=exclusive_process"

# Load necessary modules
module load VTK/9.2.6-python-3.10.14-cuda-12.6

# Activate virtual environment
source .venv/bin/activate

# Run the stability script for this specific k
python stability_sweep.py --data_path 9ng_atac_like.h5ad --output_dir ./results_stability --n_runs 10 --k ${k}
EOF

    # Submit the job
    bsub < stab_job_k${k}.bsub
    
    # Clean up the temporary job script
    rm stab_job_k${k}.bsub
    
done
