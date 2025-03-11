export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python finetune.py \
--agent mca \
--config experiments/configs/train_config.py:kitchen_sac \
--project kitchen-finetune \
--env kitchen-partial-v0 \
--reward_scale 1.0 \
--reward_bias -4.0 \
--utd 1 \
--batch_size 256 \
--warmup_steps 5000 \
--num_offline_steps 1000000 \
$@
