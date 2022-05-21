class ConfigModelArgs:
    cat_dim= 8
    img_size= 64
    z_dim=10000 # <-----
    channel_dim=128
    hidden_dim= 512
    loss_weights = {'face': 10.0,'emotion': 1.0} # <-----
    expert_type= "moe"
    dataset_path= '/home/studenti/ballerini/datasets/RAVDESS_frames'


class ConfigTrainArgs:
    learning_rate= 1e-5
    optim_betas= [ 0.95, 0.98 ]
    num_epochs= 10
    batch_size= 32
    num_workers= 40
    checkpoint_every= 20
    checkpoint_path= "./"
    save_model= True
    model_save_path= "../trained_models/ravdess_mvae_data_10k.save"
    seed= 100
    use_cuda= True
    annealing_type= "static" #static, linear, cyclical
    cyclical_annealing= {
      'min_beta': 0.0001,
      'max_beta': 0.8,
      'num_cycles': 8,
      'annealing_percentage': 0.9
    }
    linear_annealing= {
      'min_beta': 0.0001,
      'max_beta': 0.1,
    }
    static_annealing_beta= 0.00001 #<---------