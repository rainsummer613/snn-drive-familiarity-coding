#!/bin/bash

#SBATCH --array=0-399 # how many tasks in the array
#SBATCH -o .../exc_inh-%a.out

# Load software
spack load miniconda3  # module load anaconda3
source activate ...    # activate Python environment

# Run python script with a command line argument
srun python run_amn.py -c $SLURM_ARRAY_TASK_ID
