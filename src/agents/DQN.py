from typing import Any, Mapping
import torch
import torch.nn as nn
import numpy as np

from agents.utils.ExperianceBuffer import BasicExperienceBuffer

class DQN(nn.Module):
    def __init__(
            self, 
            input_size: int, 
            output_size: int, 
            lr:float=0.0001, 
            number_of_hidden_layers: int = 4,
            dense_layer_size: int=36, 
            buffer_size: int=100000, 
            batch_size: int=64, 
            gamma: float=0.99, 
            device: torch.device=torch.device("cpu")
            ):
        
        super(DQN, self).__init__()
        self.criterion = torch.nn.MSELoss()
        
        # build the input and output model layers plus a connecting ReLU layer 
        self.model_input_layer = nn.Linear(input_size, dense_layer_size)
        self.model_ReLU_layer = nn.ReLU()
        self.model_output_layer = nn.Linear(dense_layer_size, output_size)
        
        # build n hidden layers 
        self.model_layers = []
        for i in range(number_of_hidden_layers):
            self.model_layers.append(nn.Linear(dense_layer_size, dense_layer_size))
            self.model_layers.append(nn.ReLU())

        self.optimizer = torch.optim.AdamW(self.parameters(), lr)
        self.experiance_buffer = BasicExperienceBuffer(buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device
        
        self.model_name = "DQN"
        
        self.model_details = ""
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Gets the output of the model's neural network given an input

        Args:
            x (torch.Tensor): Input into the model's neural network

        Returns:
            torch.Tensor: The output of the model's neural network
        """
        # put x through the input layer
        x = self.model_input_layer(x)
        x = self.model_ReLU_layer(x)
        
        # put x through the hidden layers
        for model_layer in self.model_layers:
            x = model_layer(x)
        
        # put x through the output layer
        y = self.model_output_layer(x)
        return y

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """ Gets the output of the model's neural network given an input without gradient propagation

        Args:
            state (torch.Tensor): Input into the model's neural network

        Returns:
            torch.Tensor: The output of the model's neural network
        """
        with torch.no_grad():
            return self.forward(state)

    def hard_update_model(self, values:Mapping[str, Any]):
        """ Sets the models parameters

        Args:
            values (Mapping[str, Any]): The models new parameters
        """
        self.load_state_dict(values)

    def save_model(self, name):
        """ Saves the model to the disk under the given name

        Args:
            name (_type_): The file name
        """
        torch.save(self.state_dict(), name)
        
    def optimize_model(self, target_model: nn.Module) -> torch.Tensor:
        """ updates the model based on calculated loss

        Args:
            target_model (nn.Module): The target model used to impliment Double Q-Learning
        """
        if self.check_buffer_satisfied() is True:
            # if the buffer has the required number of samples calculate the loss
            loss = self.get_loss(target_model)
            
            # propagate the loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss
        else:
            raise(BufferError)
        
    def check_buffer_satisfied(self) -> bool:
        """ Checks if the buffer is sufficently filled for a batch to be extracted

        Returns:
            bool: True if sufficiently filled else False
        """
        if self.experiance_buffer.getLength() > self.batch_size:
            return True
        else:
            return False
    
    def get_loss(self, target_model: nn.Module) -> torch.Tensor:
        """ Calculates the loss of the model by selecting a random batch from the buffer and calculating the current and target q values

        Args:
            target_model (nn.Module): The target model used to impliment Double Q-Learning

        Returns:
            torch.Tensor: The loss of the calculated from this batch 
        """

        sample = self.experiance_buffer.getSample(amount=self.batch_size)

        states, actions, rewards, terminals, next_states = zip(*sample)
        lookup = {True: 0, False: 1}

        # convert the batch observations into tensors
        states = torch.tensor(np.array(states), dtype=torch.float).to(self.device)
        actions = torch.tensor(np.array(actions).reshape(-1, 1), dtype=torch.int64).to(self.device)
        rewards = torch.tensor(np.array(rewards).reshape(-1, 1), dtype=torch.float).to(self.device)
        terminals = torch.tensor(np.array(list(map(lambda x: lookup[x], terminals))).reshape(-1, 1), dtype=torch.int64).to(self.device) # map true false to 1, 0
        next_states = torch.tensor(np.array(next_states), dtype=torch.float).to(self.device)

        # current q value
        current_q_value = self(states).gather(1, actions)

        # expected return from next step
        # detach since we dont want gradients for the next q to propagate
        next_q_value = target_model(next_states).detach().max(dim=1, keepdim=True)[0]

        # target q value
        # a terminal state will only get the current rewards and no future rewards
        target = rewards + (self.gamma * next_q_value * terminals)

        loss = self.criterion(current_q_value, target)
        
        return loss