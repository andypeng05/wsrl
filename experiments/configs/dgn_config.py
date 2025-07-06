from ml_collections import ConfigDict

from experiments.configs import sac_config


def get_config(updates=None):
    config = sac_config.get_config()
    
    # Override SAC defaults for DGN
    config.critic_ensemble_size = 10  # Larger ensemble for Adroit tasks
    
    # DGN-specific parameters (from paper's Table 2)
    config.dgn_update_interval = 2000  # N in Algorithm 1
    config.dgn_annealing_timescale = 30000  # τ for annealing
    config.dgn_shutoff_success_threshold = None  # Can be set to 0.5 for success-based shutoff
    config.dgn_shutoff_epochs = 10  # n epochs to measure success rate
    
    # DGN covariance network architecture (from paper)
    config.covariance_network_kwargs = ConfigDict(
        {
            "hidden_dims": [256, 256],  # MLP Hidden Size from Table 2
            "dropout_rate": 0.5,  # Dropout from Table 2
        }
    )
    
    # DGN covariance optimizer (from paper's Table 2)
    config.dgn_covariance_optimizer_kwargs = ConfigDict(
        {
            "learning_rate": 1e-4,  # Same as other networks
            "weight_decay": 3e-2,  # Weight Decay from Table 2
            "optimizer": "adamw",  # AdamW from Table 2
        }
    )
    
    if updates is not None:
        config.update(ConfigDict(updates).copy_and_resolve_references())
    
    return config 