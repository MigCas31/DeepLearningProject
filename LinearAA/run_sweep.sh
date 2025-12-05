#!/bin/bash

# List of archetype counts to try (5 to 15)
ARCHETYPES=({5..15})

for k in "${ARCHETYPES[@]}"; do
    echo "Submitting job for ${k} archetypes..."
    
    # Create a temporary job script with the specific k value
    # We replace the variable placeholder with the actual number
    sed "s/\${N_ARCHETYPES:-25}/$k/g" submit_job.bsub > submit_job_temp_${k}.bsub
    
    # Submit the temporary script
    bsub -J "AA_Train_${k}" < submit_job_temp_${k}.bsub
    
    # Clean up
    rm submit_job_temp_${k}.bsub
done

echo "All jobs submitted."
