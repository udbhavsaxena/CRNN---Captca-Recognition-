import torch
from torch import nn
from torch.nn import functional as F


class CaptchaModel(nn.Module):

    def __init__(self, num_chars):
        super(CaptchaModel,self).__init__()


        self.conv_1 = nn.Conv2d(3,128, kernel_size=(3,3), padding = (1,1)) # first
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))
        self.conv_2 = nn.Conv2d(128,64, kernel_size=(3,3), padding = (1,1)) # first
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))
      
        self.linear_l = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.2)
        # self.linear_2 = nn.Linear(1152, 64)

        # adding an LSTM/GRU
        self.gru = nn.GRU(64,32, bidirectional = True, num_layers = 2,
        dropout = 0.25)
        self.output = nn.Linear(64, num_chars + 1)
        

    def forward(self,images, targets=None):

        bs,c,h,w = images.size()
        # print(bs,c,h,w)

        x = F.relu(self.conv_1(images))
        # print(x.size())
        x = self.max_pool_1(x)
        # print(x.size())
        x = F.relu(self.conv_2(x))
        # print(x.size())
        x = self.max_pool_2(x) # [1,64,18,75] --> [bs, filters, height, widhth]
        x = x.permute(0,3,1,2) # position wise permutation -- > [1,75,64,18]
        # we are doing this because we want to have a look at the width of the img
        # why? because RNN needs it 
        # print(x.size())
        x = x.view(bs,x.size(1), -1) # 
        # print('After permutation and view fn:', x.size())

        # after adding the linear layer
        x = self.linear_l(x)
        x = self.drop_1(x)
        # print('The size after linear and dropout:', x.size())

        # after adding the recurrent layer: GRU
        x, _ = self.gru(x)
        # print('The size after gru intro: ', x.size())

        # after adding linear at output
        x = self.output(x)
        # print('THe size after adding linear:', x.size()) 

        # final permutation
        x = x.permute(1,0,2)
        # print('The final permutation:', x.size())

        if targets is not None:
            # a loss function which makes sense with sequence with variable len
            # use CTC
            log_softmax_values = F.log_softmax(x, 2) # 2 for classes

            input_lengths = torch.full(
                size = (bs,), fill_value=log_softmax_values.size(0),
                dtype=torch.int32
            )
            print('Input Lengths:',input_lengths)
            targets_lengths = torch.full(
                size = (bs,), fill_value=targets.size(1),
                dtype=torch.int32
            )
            print('Target Lengths:', targets_lengths)
            loss = nn.CTCLoss(blank = 0)(
                log_softmax_values,
                targets,
                input_lengths,
                targets_lengths
            )
            return x, loss
            
        return x, None

if __name__ == '__main__':
    cm = CaptchaModel(19) #  19 diff characters
    img = torch.rand(5,3,75,300)
    target = torch.randint(1, 20,(5 ,5))
    x, loss = cm(img, target)
        

