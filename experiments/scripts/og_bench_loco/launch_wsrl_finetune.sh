export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python finetune.py \
--agent sac \
--config experiments/configs/train_config.py:ogbench_cql_redq \
--project ogbench-finetune \
--reward_scale 10.0 \
--reward_bias -5.0 \
--num_offline_steps 1_000_000 \
--env antmaze-large-navigate-v0 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
$@

