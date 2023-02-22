OMP_NUM_THREADS=8
NUM_PROC=$1
shift
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_PROC scripts/main.py "$@"
