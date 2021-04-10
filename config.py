import torch as T
import os


class Hyper:
    alpha=0.0003    # learning rate for actor network
    beta=0.0003     # learning rate for critic and value networks
    gamma = 0.99    # discount factor
    tau=0.005       # tests show that 0.005 is about the best value
    batch_size=100
    layer1_size=256
    layer2_size=256
    n_games = 500   # There are on average about 500 steps per game
    n_actions = 9
    #max_size=1000000
    max_size=1000   # 1 million is a better value but my computer can't take it
    image_shape = (84,84,1)     # resize image to improve performance
    image_jump = 4              # input every fourth image
    chkpt_dir = 'save_model'
    plots_dir = "plots"

    def init():
        # Gaurantees the folders for output exist
        if os.path.isdir(Hyper.plots_dir) == False:
            os.mkdir(Hyper.plots_dir)
        if os.path.isdir(Hyper.chkpt_dir) == False:
            os.mkdir(Hyper.chkpt_dir)

        print("\n"*3)
        print("*"*100)
        print("Hyperparameters used:")
        print("---------------------")
        print(f"environment = {Constants.env_id}")
        print(f"alpha = {Hyper.alpha}")
        print(f"beta = {Hyper.beta}")
        print(f"gamma = {Hyper.gamma}")
        print(f"tau = {Hyper.tau}")
        print(f"batch size = {Hyper.batch_size}")
        print(f"number of games = {Hyper.n_games}")
        print("*"*100)


class Constants:
    env_id = 'MsPacmanNoFrameskip-v4'
    device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')