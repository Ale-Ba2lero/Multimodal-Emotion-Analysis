class ConfigModelArgs:
    cat_dim= 8
    img_size= 64
    z_dim= 50                 #<-----2048, 4096, 8192
    num_filters= 8             #<-----32, 64, 128
    hidden_dim= 512
    loss_weights = {'face': 1.0,'emotion': 50.0} # <-----
    expert_type= "moe"
    dataset_path= '/home/studenti/ballerini/datasets/RAVDESS_frames'


class ConfigTrainArgs:
    learning_rate= 5e-4         #<--------- 1e-3 1e-4 1e-5 ### next 5e-5
    static_annealing_beta= 1e-6 #<--------- 1e-5 1e-6 1e-7
    num_epochs= 50
    batch_size= 8
    num_workers= 32
    
    checkpoint_every= 20
    optim_betas= [ 0.95, 0.98 ]
    checkpoint_path= "./"
    save_model= True
    model_save_path= "../trained_models/ravdess_mvae.save"
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
    