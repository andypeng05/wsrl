export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

# env: pen-binary-v0, door-binary-v0, relocate-binary-v0

python3 finetune.py \
--agent dgn \
--config experiments/configs/train_config.py:adroit_dgn \
--project dgn-adroit-finetune \
--num_offline_steps 0 \
--reward_scale 10.0 \
--reward_bias 5.0 \
--env pen-binary-v0 \
--utd 20 \
--batch_size 2560 \
--warmup_steps 0 \
$@ 