from ml_collections import ConfigDict


def get_config(updates=None):
    """Config for standalone DGN module."""
    config = ConfigDict()
    
    # Network architecture
    config.hidden_dims = (256, 256)
    config.dropout_rate = 0.5
    config.activation = "relu"
    
    # Training configuration
    config.learning_rate = 1e-4
    config.dgn_batch_size = 128
    config.dgn_cov_train_epochs = 10
    
    # Loss configuration
    config.dgn_entropy_coef = 0.01
    
    # Noise scheduling
    config.dgn_annealing_timescale = 30000
    
    # Success-based shutoff (optional, from paper)
    config.dgn_shutoff_success_threshold = None
    config.dgn_shutoff_epochs = 10
    
    # Numerical stability
    config.cov_diagonal_eps = 1e-5
    config.cov_max_diagonal = 5.0
    
    # Optimizer configuration (from paper's Table 2)
    config.optimizer_kwargs = ConfigDict({
        "learning_rate": 1e-4,
        "weight_decay": 3e-2,  # Weight Decay from Table 2
    })
    
    # How often to update DGN during RL training
    config.dgn_update_interval = 2000 
    
    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    
    return config 