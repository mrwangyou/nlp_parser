import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib

torch.set_num_threads(8)

# torch.manual_seed(4)
# random.seed()


class TweetDataset(Dataset):  # Done!

    def __init__(self, list):
        self.textList = []
        for num in list:
            self.textList.append(int(num))

    def __len__(self):
        return len(self.textList)

    def __getitem__(self, index):
        ID = self.textList[index]
        path = './tsr_dict/tsr_dict' + str(ID) + '.pkl'
        text = joblib.load(path)
        return text
        # text == {}
        # text['label'] == torch.tensor
        # text['text'] == torch.tensor


class LSTMTokenization(nn.Module):

    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 label_size,
                 ):
        super(LSTMTokenization, self).__init__()
        self.batch_size = 1
        self.hidden_dim = hidden_dim
        self.label_size = label_size

        # 3 lines below to deal with word embedding
        self.word_embeddings = nn.Embedding(num_embeddings=len(embed_mat), embedding_dim=len(embed_mat[0]))
        self.word_embeddings.weight.data = embed_mat
        self.word_embeddings.weight.requires_grad = False

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1)
        self.hidden = self.init_hidden()
        self.linear = nn.Linear(hidden_dim, label_size)

    def init_hidden(self):
        return (
            autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda()),
            autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim).cuda())
        )

    def forward(self, text):
        self.batch_size = len(text[0][0])
        # print('##' + str(text.shape))
        self.hidden = self.init_hidden()
        # print('!' + str(text.shape))
        embeds = self.word_embeddings(text)
        # lstm_in = embeds.view(len(text), self.batch_size, -1)
        lstm_in = embeds
        # print(lstm_in.squeeze(0).shape)
        lstm_out, self.hidden = self.lstm(lstm_in.squeeze(0), self.hidden)
        y = self.linear(lstm_out)
        y = y.transpose(1, 2)
        log_probs = F.log_softmax(y, dim=1)
        print('##')
        print(log_probs.shape)
        # print(log_probs)
        return log_probs


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / (len(truth) + 1)


def train():
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    EPOCH = 1

    # fullDataset = TweetDataset('./PTH/accountList.txt')
    # trSize = int(len(fullDataset) * 0.7)
    # devSize = int(len(fullDataset) * 0.2)
    # testSize = len(fullDataset) - trSize - devSize
    # trainDataset, devDataset, testDataset = torch.utils.data.random_split(fullDataset, [trSize, devSize, testSize])
    trainLoader = DataLoader(dataset=TweetDataset(range(2004)),
                             shuffle=True,
                             )

    devLoader = DataLoader(dataset=TweetDataset(range(2004, 2290)),
                           shuffle=True
                           )

    testLoader = DataLoader(dataset=TweetDataset(range(2290, 2863)),
                            shuffle=True
                            )

    model = LSTMTokenization(embedding_dim=EMBEDDING_DIM,
                             hidden_dim=HIDDEN_DIM,
                             label_size=9,
                             )

    # 3 lines below put the code on the GPU
    device_ids = [0, 1, 2, 3]
    model = nn.DataParallel(model, device_ids=device_ids)
    model.cuda()

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #    optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9)
    bestTillNow = 0
    for i in range(EPOCH):
        print('epoch: %d start!' % i)
        train_epoch(model, trainLoader, loss_function, optimizer)
        dev_acc = evaluate_epoch(model, devLoader, loss_function, 1)
        test_acc = evaluate_epoch(model, testLoader, loss_function, 2)
        if dev_acc > bestTillNow:
            bestTillNow = dev_acc
            print('New best model! Acc = ' + str(dev_acc))
            torch.save(model.module.state_dict(), './bestModel/Epoch' + str(i) + 'acc' + str(test_acc) + '.pt')


def train_epoch(model,
                train_data,
                loss_function,
                optimizer,
                ):
    model.train()
    avg_loss = 0
    acc = 0
    cnt = 0
    sumLoss = 0
    num2Backward = 16
    for batch in train_data:
        cnt = cnt + 1
        # if cnt % 100 == 0:
        # print('Begin ' + str(int(cnt / 100)) + ' batch in an epoch!')
        label = batch['label'].squeeze(0).long().cuda()
        text = batch['text'].cuda()
        # label = label.cuda()
        # text = text.cuda()
        pred = model(text)
        # if pred[0][0] > pred[0][1] and label == 0 or pred[0][0] < pred[0][1] and label == 1:
        #     acc = acc + 1
        # print('?')
        # print(label.shape)
        loss = loss_function(pred, label)
        avg_loss += loss.item()
        sumLoss = sumLoss + loss
        model.zero_grad()
        if cnt % num2Backward == 0:
            # print('Acc now: ' + str(acc / cnt))

            sumLoss = sumLoss / float(num2Backward)
            sumLoss.backward()
            sumLoss = 0
            optimizer.step()
    avg_loss = avg_loss / cnt
    # acc = acc / float(cnt)
    print('train: ')
    print('loss: ' + str(avg_loss))
    # print('acc: ' + str(acc))


def evaluate_epoch(model,
                   train_data,
                   loss_function,
                   ii,
                   ):
    model.eval()
    acc = 0
    cnt = 0
    for batch in train_data:
        cnt = cnt + 1
        label = batch['label'].squeeze(0).long().cuda()
        text = batch['text'].cuda()
        pred = model(text)
        # if pred[0][0] > pred[0][1] and label == 0 or pred[0][0] < pred[0][
        #     1] and label == 1:  # 0 for bot and 1 for human
        #     acc = acc + 1
        model.zero_grad()

        # loss = loss_function(pred, label)
    # acc = acc / float(cnt)
    if ii == 1:
        print('val: ')
    else:
        print('test: ')
    # print('acc: ' + str(acc))

    return acc


embed_mat = torch.load('./embedding.pth')
train()
