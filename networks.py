import torch
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class V2V(torch.nn.Module):

    def __init__(self, input_dim, n_epochs, batch_size):
        super(V2V, self).__init__()
        self.fc = torch.nn.Sequential(torch.nn.Linear(input_dim, input_dim), torch.nn.GELU(), torch.nn.Linear(input_dim, input_dim)).to(device)
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.Adam(self.fc.parameters())
        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):
        return self.fc(x)
    
    def train(self, training_features, train_labels):
        for epoch in range(self.n_epochs):
            order = torch.randperm(len(training_features))
            loss=0
            for start_index in range(0, len(training_features), self.batch_size):
                self.optimizer.zero_grad()
        
                batch_indexes = order[start_index:start_index+self.batch_size]
        
                X_batch = training_features[batch_indexes].to(device)
                y_batch = train_labels[batch_indexes].to(device)
        
                preds = self.fc(X_batch) 
                
                loss_value = self.loss_func(preds, y_batch)
                loss_value.backward()

                self.optimizer.step()
                
                loss+=loss_value.item()
            print(loss)