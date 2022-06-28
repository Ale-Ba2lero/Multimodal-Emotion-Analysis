class ConfigModelArgs:
    cat_dim= 8
    img_size= 64
    z_dim= 50
    num_filters= 32
    hidden_dim= 256
    loss_weights = {'face': 1.0,'emotion': 1e6}
    expert_type= "poe" # moe, poe, fusion
    dataset_path= '/home/studenti/ballerini/datasets/RAVDESS_frames_dsl'
    
    image_feature_size = 64
    emotion_feature_size = 64

class ConfigTrainArgs:
    learning_rate= 5e-6         #<--------- 1e-3 1e-4 1e-5
    static_annealing_beta= 1e-7
    num_epochs= 100
    batch_size= 32
    num_workers= 32
    optim_betas= [ 0.95, 0.98 ]
    seed= 100
    use_cuda= True
    checkpoint_every = 20
    checkpoint_save_path = "../trained_models/mmvae_fusion_checkpoint.save"
    