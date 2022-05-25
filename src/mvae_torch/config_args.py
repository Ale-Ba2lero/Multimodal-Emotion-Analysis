class ConfigModelArgs:
    cat_dim= 8
    img_size= 64
    z_dim=8192 # <----- 8192, 10000
    channel_dim= 128
    hidden_dim= 512
    loss_weights = {'face': 1.0,'emotion': 5.0} # <----- *
    expert_type= "moe"
    dataset_path= '/home/studenti/ballerini/datasets/RAVDESS_frames_ds'


class ConfigTrainArgs:
    learning_rate= 5e-6
    optim_betas= [ 0.95, 0.98 ]
    num_epochs= 100
    batch_size= 128
    num_workers= 20
    checkpoint_every= 20
    checkpoint_path= "./"
    save_model= True
    model_save_path= "../trained_models/ravdess_mvae_data_8k_altloss_06.save"
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
    static_annealing_beta= 0.000005 #<---------