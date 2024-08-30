import subprocess
import os
import shlex
from pathlib import Path

wd: str = Path(__file__).parent.absolute()
os.environ["PYTHONPATH"] = f"{wd}:{os.getenv('PYTHONPATH', '')}"
script: str = os.path.join(wd, "flux_example.py")
model_id = "black-forest-labs/FLUX.1-schnell"
inference_step = 4
height = 1024
width = 1024
task_args = f"--height {height} --width {width}"
n_gpus = 1
parallel_args = f"--ulysses_degree {n_gpus}"
compile_flag = "--use_torch_compile"

command: str = f"""
                torchrun --nproc_per_node={n_gpus} {script}
                --model {model_id}
                --warmup_steps 3
                {compile_flag}
                {parallel_args}
                {task_args}
                --prompt "A small dog"
                --no_use_resolution_binning
                --num_inference_steps {inference_step}
                """
print(command)
print(shlex.split(command))
subprocess.run(shlex.split(command))
