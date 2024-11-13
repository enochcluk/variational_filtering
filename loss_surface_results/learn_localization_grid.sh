#!/bin/bash

# Define parameter arrays
models=("l96" "ks")
num_steps_list=(50 100)
initializations=("sin" "cos" "random")
n_ensembles=(10 20)

# Create a directory for logs if it doesn't exist
mkdir -p logs

# Loop over all combinations of parameters
for model in "${models[@]}"; do
    for num_steps in "${num_steps_list[@]}"; do
        for initialization in "${initializations[@]}"; do
            for n_ensemble in "${n_ensembles[@]}"; do
                # Construct a unique job name based on parameters
                job_name="${model}_${initialization}_steps${num_steps}_ensemble${n_ensemble}"

                # Write the SBATCH script - be sure to note the directory location of the python script
                sbatch --job-name="$job_name" \
                    --time=10:00:00\
                    --output="logs/${job_name}.out" \
                    --error="logs/${job_name}.err" \
                    --wrap="python learn_distances.py --model $model --num_steps $num_steps --initialization $initialization --n_ensemble $n_ensemble"
                
                echo "Submitted job: $job_name"
            done
        done
    done
done
