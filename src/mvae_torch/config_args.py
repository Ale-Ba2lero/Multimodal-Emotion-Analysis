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
    
    z_dim = 25
    hidden_dim= 128
    
    au_weight= 10
    emotion_weight= 0.001
    
    modes = {'au': True,  
             'face': None, 
             'emotion': True
            }
    
    expert_type = "poe" # moe, poe,W fusion
    dataset_path = '/home/studenti/ballerini/datasets/au-emo_2.csv'

class ConfigTrainArgs:
    batch_size= 256
    learning_rate= 0.0001 
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
BEST POE 55-92 
L2 - L2
{'cat_dim': 8, 'au_dim': 18, 'latent_space_dim': 25, 'hidden_dim': 128, 'num_filters': 32, 'modes': {'au': True, 'face': None, 'emotion': True}, 'au_weight': 10, 'emotion_weight': 0.001, 'expert_type': 'poe', 'use_cuda': True}
{'learning_rate': 0.0001, 'beta':1e-6, 'optim_betas': [0.95, 0.98], 'num_epochs': 50, 'batch_size': 256}


Best trial config: {'batch_size': 128, 'z_dim': 25, 'hidden_dim': 512, 'lr': 0.001, 'alpha': 0.1, 'beta': 0.1, 'au_weight': 45751.87811347376, 'emotion_weight': 0.002556536731523064}
Best trial final validation loss: 34.24396201441109
Best trial final validation accuracy: 0.7548881913291786
'''