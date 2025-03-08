export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python finetune.py \
--agent iql \
--config experiments/configs/train_config.py:ogbench_iql \
--project ogbench-finetune \
--reward_scale 1.0 \
--reward_bias 0.0 \
--num_offline_steps 1_000_000 \
--env cube-single-play-v0 \
$@
