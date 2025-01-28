export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python finetune.py \
--agent sac \
--config experiments/configs/train_config.py:ogbench_wsrl \
--project ogbench-finetune \
--num_offline_steps 1_000_000 \
--env cube-single-play-v0 \
--reward_scale 1.0 \
--reward_bias 0.0 \
--utd 4 \
--batch_size 1024 \
--warmup_steps 5000 \
$@
