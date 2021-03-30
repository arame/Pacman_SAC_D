import torch as T

class Hyper:
    alpha=0.0003
    beta=0.0003
    reward_scale=2
    tau=0.005
    batch_size=256
    layer1_size=256
    layer2_size=256
    n_games = 250
    n_actions=2
    #max_size=1000000
    max_size=1000
    image_shape = (84,84,1)
    image_jump = 4

class Constants:
    env_id = 'MsPacmanNoFrameskip-v4'
    chkpt_dir='save_model/sac'
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')