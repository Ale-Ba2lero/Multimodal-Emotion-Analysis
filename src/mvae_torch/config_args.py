class ConfigModelArgs:
    cat_dim=8
    img_size= 64
    z_dim= 256
    hidden_dim= 512
    loss_weights = {'face': 1.0,'emotion': 1.0}
    expert_type= "moe"
    dataset_path= '/home/studenti/ballerini/datasets/RAVDESS_frames_ds'


class ConfigTrainArgs:
    learning_rate= 0.0001
    optim_betas= [ 0.95, 0.98 ]
    num_epochs= 10
    batch_size= 32
    checkpoint_every= 20
    checkpoint_path= "./"
    save_model= True
    model_save_path= "trained_models/ravdess_mmvae_pretrained_plain.pt",
    stats_save_path= "trained_models/ravdess_mmvae_pretrained_plain_stats.pt",
    seed= 100
    use_cuda= True
    annealing_type= "static"
    cyclical_annealing= {
      'min_beta': 0.0001,
      'max_beta': 0.8,
      'num_cycles': 8,
      'annealing_percentage': 0.9
    }
    linear_annealing= {
      'min_beta': 0.001,
      'max_beta': 1.0,
    }
    static_annealing_beta= 0.1