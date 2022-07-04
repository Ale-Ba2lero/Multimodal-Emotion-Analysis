class Mode:
    def __init__(self, weight:float, feature_size:int):
        self.weight = weight
        self.feature_size = feature_size
    

class ConfigModelArgs:
    cat_dim = 8
    au_dim = 18
    
    # conv network args 
    img_size = 64
    num_filters = 32
    hidden_dim= 128
    z_dim = 50
    
    modes = {'au': Mode(1e5, 32),  
             'face': None, 
             'emotion': Mode(1.0, 32)
            }
    
    expert_type = "poe" # moe, poe, fusion
    dataset_path = '/home/studenti/ballerini/datasets/au-emo.csv'

class ConfigTrainArgs:
    learning_rate= 1e-3        #<--------- 1e-3 1e-4 1e-5
    static_annealing_beta= 1e-6
    num_epochs= 25
    batch_size= 512
    num_workers= 32
    optim_betas= [ 0.95, 0.98 ]
    seed= 100
    use_cuda= True
    checkpoint_every = None
    checkpoint_save_path = "../trained_models/mmvae_fusion_checkpoint.save"