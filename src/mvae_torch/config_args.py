class Mode:
    def __init__(self, weight:float, feature_size:int):
        self.weight = weight
        self.feature_size = feature_size
    

class ConfigModelArgs:
    cat_dim = 8
    au_dim = 17
    z_dim = 25
    hidden_dim= 128
    au_weight= 10
    emotion_weight= 0.001
    expert_type = "poe" # moe, poe, fusion
    modes = {'au': True,  
             'face': None, 
             'emotion': True}
    
    dataset_path = '/home/studenti/ballerini/Multimodal_RSA/src/util/au-emo_2.csv'
    # conv network args 
    num_filters = 32
    img_size = 64

class ConfigTrainArgs:
    learning_rate= 0.0001
    batch_size= 256
    alpha=1.0
    static_annealing_beta= 1e-6
    num_epochs= 50
    num_workers= 32
    optim_betas= [ 0.95, 0.98 ]
    seed= 100
    use_cuda= True
    checkpoint_every = None
    checkpoint_save_path = "../trained_models/mmvae_fusion_checkpoint.save"
    
'''
Best trial config: {'batch_size': 512, 'z_dim': 25, 'hidden_dim': 512, 'lr': 0.0001, 'alpha': 0.5, 'beta': 0.0001, 'au_weight': 49492.5907103356, 'emotion_weight': 0.0018254361473506392}

BEST POE 55-92 
L2 - L2
{'cat_dim': 8, 'au_dim': 17, 'latent_space_dim': 25, 'hidden_dim': 128, 'num_filters': 32, 'modes': {'au': True, 'face': None, 'emotion': True}, 'au_weight': 10, 'emotion_weight': 0.001, 'expert_type': 'poe', 'use_cuda': True}
{'learning_rate': 0.0001, 'alpha'=1.0, 'beta':1e-6, 'optim_betas': [0.95, 0.98], 'num_epochs': 50, 'batch_size': 256}
'''