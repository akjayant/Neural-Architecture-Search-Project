import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import Lunar_pytorch_enas
from pexecute.process import ProcessLoom

def detach(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(detach(v) for v in h)

def get_variable(inputs, cuda=False, **kwargs):
    if type(inputs) in [list, np.ndarray]:
        inputs = torch.Tensor(inputs)
    if cuda:
        out = Variable(inputs.cuda(), **kwargs)
    else:
        out = Variable(inputs, **kwargs)
    return out

class Controller(nn.Module):
    def __init__(self,args):
        super(Controller,self).__init__()
        self.args = args
        self.decoder_order = []
        for i in range(self.args['n_blocks']):
                self.decoder_order.extend([self.args['n_dense_layer'],self.args['n_activation_functions']])
        num_total_tokens = sum(self.decoder_order)
        self.encoder = torch.nn.Embedding(num_total_tokens,
                                          args['controller_hid'])
        self.lstm = torch.nn.LSTMCell(args['controller_hid'], args['controller_hid'])
        self.decoders = []
        for idx, size in enumerate(self.decoder_order):
            decoder = torch.nn.Linear(args['controller_hid'], size)
            self.decoders.append(decoder)
        self._decoders = torch.nn.ModuleList(self.decoders)
        #self.reset_parameters()
        self.static_init_hidden = self.init_hidden(1,self.args['controller_hid'])
        self.static_input = get_variable(torch.zeros(1,self.args['controller_hid']),False)

    def forward(self,inputs,hidden,block_idx,is_embed):
        if is_embed == False:
            embedded_input = self.encoder(inputs)
        else:
            embedded_input = inputs

        h,c = self.lstm(embedded_input,hidden)
        #trigger the required decoder
        logits = self.decoders[block_idx](h)

        return logits, (h,c)






    def init_hidden(self, batch_size,hidden_size):
        #zeros = torch.zeros(1,hidden_size)
        #return (torch.tensor(zeros),
       #         torch.tensor(zeros.clone()))

        zeros = torch.zeros(batch_size, hidden_size)
        return (get_variable(zeros, False, requires_grad=False),
                get_variable(zeros.clone(), False, requires_grad=False))

    def sample(self):
        activations = []
        dense_layer_type = []
        inputs = self.static_input
        hidden = self.static_init_hidden
        for i in range(2*self.args['n_blocks']):
            logits, hidden = self.forward(inputs,hidden,i,i==0)
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            action = probs.multinomial(num_samples=1).data
            #print(action)
            mode = i % 2
            inputs = get_variable(
                 action[:, 0] ,
                 requires_grad=False)

            if mode == 0:
                 activations.append(action[:, 0])
            elif mode == 1:
                 dense_layer_type.append(action[:, 0])

        #print(activations,dense_layer_type,log_prob)
        return [activations,dense_layer_type,log_prob]


#-------------------ENAS DRIVER---------------------------

class ENAS():
    def __init__(self,args,epochs):
        self.controller_model = Controller(args)
        self.epochs = epochs
        self.baseline = None
        self.controller_optim = torch.optim.Adam(self.controller_model.parameters(),lr=.035)
        self.samples_per_policy = 4
    def train(self):
        reward_history = []
        adv_history = []
        for ep in range(self.epochs):
            print("---------------EPOCH--"+str(ep)+"---------------")
            samples = []
            for m in range(self.samples_per_policy):
                s = self.controller_model.sample()
                n_fc1 = dense_layer_list[s[0][0].item()]
                n_fc2 = dense_layer_list[s[0][1].item()]
                af_1 = s[1][0].item()
                af_2 = s[1][1].item()
                samples.append([n_fc1,n_fc2,af_1,af_2,0])
            #print(n_fc1,n_fc2,af_1,af_2)
            print("Reference - ")
            print(activation_function_list)
            print(dense_layer_list)
            print("Training architecture -",samples)
            #time_taken = Lunar_pytorch_enas.train_dqn(n_fc1,n_fc2,af_1,af_2,0)
            print("-----------Training 4 architectures in parallel for a given policy----------------")
            loom = ProcessLoom(max_runner_cap = 4)
            loom.add_function(Lunar_pytorch_enas.train_dqn,samples[0],{})
            loom.add_function(Lunar_pytorch_enas.train_dqn,samples[1],{})
            loom.add_function(Lunar_pytorch_enas.train_dqn,samples[2],{})
            loom.add_function(Lunar_pytorch_enas.train_dqn,samples[3],{})

            output =  loom.execute()
            for m in range(self.samples_per_policy):
                a,b,c,d = samples[m]
                reward_i = Lunar_pytorch_enas.test(a,b,c,d)
                reward_history.append(reward_i)
                print("Reward = ",reward_i)


                if self.baseline is None:
                    self.baseline = reward_i
                else:
                    decay = 0.95
                    self.baseline = decay * self.baseline + (1 - decay) * reward_i


                adv = reward_i - self.baseline
                adv_history.append(adv)

                # policy loss
                log_probs = s[2]
                loss = -log_probs*get_variable(torch.tensor(float(adv)),
                                                     requires_grad=True)


                loss = loss.sum()  # or loss.mean()

            # update
            loss  = loss/self.samples_per_policy
            self.controller_optim.zero_grad()
            loss.backward()
            self.controller_optim.step()
            print(loss)

if __name__ == "__main__":
    args = {'n_dense_layer':5,'n_activation_functions':2,'controller_hid':100,'n_blocks':2}
    activation_function_list = ['ReLU','sigmoid']
    dense_layer_list = [64,128,256,1024,2048]
    e = ENAS(args,50)
    e.train()
