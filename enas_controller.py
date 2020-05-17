import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import Lunar_pytorch_enas
#from pexecute.process import ProcessLoom
from multiprocessing import Pool
import multiprocessing as mp
import pickle


def load_pickle(pkl_name):
    readfile = open(pkl_name, 'rb')
    model = pickle.load(readfile)
    return model

def dump_pickle(obj,name):
    file = open(name+str('.pkl'), 'wb')
    pickle.dump(obj, file)






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
        log_probs = []
        for i in range(2*self.args['n_blocks']):
            logits, hidden = self.forward(inputs,hidden,i,i==0)
            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)
            action = probs.multinomial(num_samples=1).data
            selected_log_prob = log_prob.gather(1,get_variable(action,requires_grad=False))
            mode = i % 2
      # 'mode' controls whether you choose dense layer type or activation function type
      # We have created a lookup embedding table of 12*100  , dense selection has 4 choices so it will choose from first 4 rows otheriwise if activation choice
      # i.e, 2 choices , it will choose from 5 or 6. (yes rest of 6 rows are useless :p)
            inputs = get_variable(
                 action[:, 0] + sum(self.decoder_order[:mode]),
                 requires_grad=False)

            if mode == 0:
                 dense_layer_type.append(action[:, 0])
            elif mode == 1:
                 activations.append(action[:, 0])
            log_probs.append(selected_log_prob[:,0])

        #print(activations,dense_layer_type,log_prob)
        return [dense_layer_type,activations,torch.cat(log_probs)]


#-------------------ENAS DRIVER---------------------------

class ENAS():
    def __init__(self,args,epochs):
        self.controller_model = Controller(args)
        self.epochs = epochs
        self.baseline = None
        self.controller_optim = torch.optim.Adam(self.controller_model.parameters(),lr=.0085)
        self.samples_per_policy = 5
    def train(self):
        convergence_count = 0
        loss_tracker = []
        reward_history = [0]
        adv_history = [0]
        for ep in range(self.epochs):
            import Lunar_pytorch_enas
            print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<---------------EPOCH--"+str(ep)+"-------------------------------->>>>>>>>>>>>>>>>>>>>>>>")
            samples = []
            for m in range(self.samples_per_policy):
                s = self.controller_model.sample()
                #print(s)
                n_fc1 = dense_layer_list[s[0][0].item()]
                n_fc2 = dense_layer_list[s[0][1].item()]
                af_1 = s[1][0].item()
                af_2 = s[1][1].item()
                samples.append((n_fc1,n_fc2,af_1,af_2,0))
            #print(n_fc1,n_fc2,af_1,af_2)
            print("Reference - ")
            print(activation_function_list)
            print(dense_layer_list)
            print("Training architecture -",samples)
            #time_taken = Lunar_pytorch_enas.train_dqn(n_fc1,n_fc2,af_1,af_2,0)
            print("-----------Training 5 architectures in parallel for a given policy----------------")
            processes = []
            mp.set_start_method("spawn",force=True)
            with Pool(self.samples_per_policy) as p:
                p.starmap(Lunar_pytorch_enas.train_dqn,samples)

            p.close()
            output_rewards = []
            with Pool(self.samples_per_policy) as p:
                output = p.starmap(Lunar_pytorch_enas.test,samples)

            p.close()
            reward_epoch = output


            epoch_average_reward_tracker = []
            for m in range(self.samples_per_policy):
                #a,b,c,d,_ = samples[m]
                reward_i = reward_epoch[m]
                epoch_average_reward_tracker.append(reward_i*30)
                reward_history.append(reward_i)
                print("Reward = ",reward_i)
                #exponential moving average of previous rewards
                if self.baseline is None:
                    self.baseline = reward_i
                else:
                    decay = 0.999
                    self.baseline = (decay) * (self.baseline) + (1-decay)*reward_i
                adv = reward_i - self.baseline
                adv_history.append(adv)
                log_probs = s[2]
                loss = -log_probs*get_variable(torch.tensor(float(adv)),requires_grad=True)
                loss = loss.sum()

            
            loss  = loss/self.samples_per_policy
            self.controller_optim.zero_grad()
            loss.backward()
            self.controller_optim.step()
            f = open("controller_performance_tracker_run_2.txt","a")
            f.write(str(sum(epoch_average_reward_tracker)/len(epoch_average_reward_tracker))+'\n')
            f.close()
            loss_tracker.append(loss)
            if ep>25:
                prev_10_losses = loss_tracker[-10:]
                if sum(prev_10_losses)/10 == 0:
                    break
            print("Contoller Loss=",loss)
        print("Best model : Dense layers - {} Activation Function - {}".format(self.controller_model.sample()[0],self.controller_model.sample()[1]))

if __name__ == "__main__":
    args = {'n_dense_layer':5,'n_activation_functions':2,'controller_hid':100,'n_blocks':2}
    activation_function_list = ['ReLU','sigmoid']
    dense_layer_list = [64,128,256,1024,2048]
    e = ENAS(args,50)
    e.train()
    dump_pickle(e,"50_iters_enas")
