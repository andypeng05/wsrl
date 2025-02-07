export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl

python finetune.py \
--agent sac \
--config experiments/configs/train_config.py:ogbench_sac \
--project ogbench-finetune \
--num_offline_steps 1000000 \
--env cube-single-play-v0 \
--config.agent_kwargs.critic_subsample_size 2 \
--reward_scale 1.0 \
--reward_bias 0.0 \
--utd 1 \
--batch_size 256 \
--warmup_steps 5000 \
--config.agent_kwargs.bc_loss_weight 0.5 \
$@
