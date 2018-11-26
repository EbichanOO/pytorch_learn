import random
import torch
from torch import nn, optim
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
    
    def forward(self, inputs, hidden):
        embedded = self.embedding(inputs).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
    
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, inputs, hidden):
        output = self.embedding(inputs).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

def train(input_tensor, target_tensor, encoder, decoder, encoder_opt, decoder_opt, criterion, max_length=MAX_LENGTH):
    teacher_forcing_ratio = 0.5
    encoder_hidden = encoder.initHidden()
    
    encoder_opt.zero_grad()
    decoder_opt.zero_grad()

    input_length = input_tensor.size(0)
    print(input_length)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss=0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        print(encoder_output[0, 0].size())
        encoder_outputs[ei] = encoder_output[0]
    
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[di])
        decoder_input = target_tensor[di]
    
    loss.backward()

    encoder_opt.step()
    decoder_opt.step()
    
    return loss.item()/target_length

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    import time
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs)) for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)




if __name__=='__main__':
    hidden_size = 256
    from dataEdit import DataSet
    DS = DataSet()
    dicts = DS.readDict()
    encoder1 = EncoderRNN(len(dicts), hidden_size).to(device)
    decoder1 = DecoderRNN(hidden_size, len(dicts)).to(device)
    
    in_datas = DS.readDatasetNum()
    print(in_datas[1])
    input_tensor = torch.tensor(in_datas[:40], dtype=torch.int)
    target_tensor = input_tensor
    for i in range(1, len(input_tensor)):
        target_tensor[i-1] = input_tensor[i]
    #input_tensor = input_tensor[i for i in range(len(input_tensor))]

    learning_rate=0.01
    encoder_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder1.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss()

    for i in range(100):
        loss = train(input_tensor, target_tensor, encoder1, decoder1, encoder_optimizer, decoder_optimizer, criterion)
        print(loss)

    torch.save(encoder1.state_dict(), 'datas/encoderM')
    torch.save(decoder1.state_dict(), 'datas/decoderM')