#!/bin/sh
#SBATCH --job-name=myChatGPT
#SBATCH --partition=quadgpu
#SBATCH --gres=gpu:2
#SBATCH --time=15-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kend4r@hotmail.com     # e-mail notification
#SBATCH --output=job_exchange_rate%j.out          # if --error is absent, includes also the errors
#SBATCH --mem=750G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10

echo "------------------------------------------------------------------------------"
echo "hostname                     =   $(hostname)"
echo "SLURM_JOB_NAME               =   $SLURM_JOB_NAME"
echo "SLURM_SUBMIT_DIR             =   $SLURM_SUBMIT_DIR"
echo "SLURM_JOBID                  =   $SLURM_JOBID"
echo "SLURM_JOB_ID                 =   $SLURM_JOB_ID"
echo "SLURM_NODELIST               =   $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST           =   $SLURM_JOB_NODELIST"
echo "SLURM_TASKS_PER_NODE         =   $SLURM_TASKS_PER_NODE"
echo "SLURM_JOB_CPUS_PER_NODE       =   $SLURM_JOB_CPUS_PER_NODE"
echo "SLURM_TOPOLOGY_ADDR_PATTERN  =   $SLURM_TOPOLOGY_ADDR_PATTERN"
echo "SLURM_TOPOLOGY_ADDR          =   $SLURM_TOPOLOGY_ADDR"
echo "SLURM_CPUS_ON_NODE           =   $SLURM_CPUS_ON_NODE"
echo "SLURM_NNODES                 =   $SLURM_NNODES"
echo "SLURM_JOB_NUM_NODES          =   $SLURM_JOB_NUM_NODES"
echo "SLURMD_NODENAME              =   $SLURMD_NODENAME"
echo "SLURM_NTASKS                 =   $SLURM_NTASKS"
echo "SLURM_NPROCS                 =   $SLURM_NPROCS"
echo "SLURM_MEM_PER_NODE           =   $SLURM_MEM_PER_NODE"
echo "SLURM_PRIO_PROCESS           =   $SLURM_PRIO_PROCESS"
echo "------------------------------------------------------------------------------"

# USER Commands

# special commands for openmpi/intel
#module load openmpi/intel-opa/gcc/64
#module load openmpi/gcc/64/4.1.2
#module load gcc/11.2.0


python myChatGPT.py


# end of the USER commands
