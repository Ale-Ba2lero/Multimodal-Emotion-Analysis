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
    
    z_dim = 50
    hidden_dim= 512
    au_weight= 10000.0
    emotion_weight= 0.01
    
    modes = {'au': True,  
             'face': None, 
             'emotion': True
            }
    
    expert_type = "poe" # moe, poe, fusion
    dataset_path = '/home/studenti/ballerini/datasets/au-emo.csv'

class ConfigTrainArgs:
    batch_size= 256
    learning_rate= 0.001        #<--------- 1e-3 1e-4 1e-5
    static_annealing_beta= 1e-6
    num_epochs= 30
    num_workers= 32
    optim_betas= [ 0.95, 0.98 ]
    seed= 100
    use_cuda= True
    checkpoint_every = None
    checkpoint_save_path = "../trained_models/mmvae_fusion_checkpoint.save"
    
'''
BEST POE 54-79 
L2 - L2
{'cat_dim': 8, 'au_dim': 18, 'latent_space_dim': 25, 'hidden_dim': 128, 'num_filters': 32, 'modes': {'au': True, 'face': None, 'emotion': True}, 'au_weight': 10, 'emotion_weight': 0.001, 'expert_type': 'poe', 'use_cuda': True}
{'learning_rate': 0.0001, 'optim_betas': [0.95, 0.98], 'num_epochs': 50, 'batch_size': 256}
'''

#{'batch_size': 256, 'latent_space_dim': 50, 'hidden_dim': 512, 'lr': 0.001, 'beta': 1e-06, 'au_weight': 10000.0, 'emotion_weight': 0.01}