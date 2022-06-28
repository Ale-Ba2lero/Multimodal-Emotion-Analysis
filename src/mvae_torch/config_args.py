class Mode:
    def __init__(self, weight:float, feature_size:int):
        self.weight = weight
        self.feature_size = feature_size
    

class ConfigModelArgs:
    cat_dim = 8
    au_dim = 18
    img_size = 64
    z_dim = 50
    num_filters = 32
    hidden_dim= 256
    modes = {'au': Mode(1.0, 64), 'face': None,'emotion': Mode(1.0, 64)}
    expert_type = "poe" # moe, poe, fusion
    dataset_path = '/home/studenti/ballerini/datasets/au-emo.csv'

class ConfigTrainArgs:
    learning_rate= 5e-6         #<--------- 1e-3 1e-4 1e-5
    static_annealing_beta= 1e-7
    num_epochs= 10
    batch_size= 32
    num_workers= 32
    optim_betas= [ 0.95, 0.98 ]
    seed= 100
    use_cuda= True
    checkpoint_every = 20
    checkpoint_save_path = "../trained_models/mmvae_fusion_checkpoint.save"