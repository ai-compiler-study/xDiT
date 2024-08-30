export PYTHONPATH=$PWD:$PYTHONPATH

export MODEL_TYPE="Flux"

export N_GPUS=1
export CFG_ARGS="--use_cfg_parallel"

export SCRIPT=flux_example.py
export MODEL_ID="black-forest-labs/FLUX.1-schnell"
export INFERENCE_STEP=4
# Flux does not apply cfg
export CFG_ARGS=""

mkdir -p ./results

for HEIGHT in 1024
do
  TASK_ARGS="--height $HEIGHT \
          --width $HEIGHT \
          --no_use_resolution_binning \
          "

  # Flux only supports SP, do not set the pipefusion degree
  export PARALLEL_ARGS="--ulysses_degree $N_GPUS"
  export COMPILE_FLAG="--use_torch_compile"

  torchrun --nproc_per_node=$N_GPUS $SCRIPT \
  --model $MODEL_ID \
  $PARALLEL_ARGS \
  $TASK_ARGS \
  --num_inference_steps $INFERENCE_STEP \
  --warmup_steps 3 \
  --prompt "A small dog" \
  $CFG_ARGS \
  $PARALLLEL_VAE \
  $COMPILE_FLAG
done