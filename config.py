
batch_size = 64
outer_lr = 0.01  # outer loop learning rate
dropout_prob = 0
num_epochs = 100
max_patience = 10 



# For compass dataset the number of input features is 8
input_dim = 8
hidden_dim = [8]
output_dim = 1

n_step = 1
n_iter = 10 #number of innner loop iterations 

inner_args = {
    'lr': 0.005     # Learning rate for inner-loop optimization
}

meta_args = {
    'meta_train': True,     # Whether to train the model in meta-learning mod            
    'num_iter': 1           # num
}

