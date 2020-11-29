# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1 2 3"

import time
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import random
START_TAG = "<START>"
STOP_TAG = "<STOP>"


# torch.manual_seed(1)

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else to_ix['<OUT>'] for w in seq]
    return torch.tensor(idxs, dtype=torch.long).cuda(1)


def prepare_sequence_batch(data, word_to_ix, tag_to_ix):
    seqs = [i[0] for i in data]
    tags = [i[1] for i in data]
    max_len = max([len(seq) for seq in seqs])
    seqs_pad = []
    tags_pad = []
    for seq, tag in zip(seqs, tags):
        seq_pad = seq + ['<PAD>'] * (max_len - len(seq))
        tag_pad = tag + ['<PAD>'] * (max_len - len(tag))
        seqs_pad.append(seq_pad)
        tags_pad.append(tag_pad)
    idxs_pad = torch.tensor([[word_to_ix[w] for w in seq] for seq in seqs_pad], dtype=torch.long)
    tags_pad = torch.tensor([[tag_to_ix[t] for t in tag] for tag in tags_pad], dtype=torch.long)
    return idxs_pad.cuda(1), tags_pad.cuda(1)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
           torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


def log_add(args):
    return torch.log(torch.sum(torch.exp(args), axis=0))


class BiLSTM_CRF_MODIFY_PARALLEL(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF_MODIFY_PARALLEL, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))


        self.transitions.data[tag_to_ix[START_TAG], :] = -100000000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -100000000
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        begin = time.time()
        init_alphas = torch.full((1, self.tagset_size), -100000000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        forward_var = init_alphas
        begin = time.time()
        print('# Begin')
        for feat in feats:
            print('# feat')
            alphas_t = []  
            for next_tag in range(self.tagset_size):
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = (forward_var + trans_score + emit_score)
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _forward_alg_new(self, feats):
        init_alphas = torch.full([self.tagset_size], -100000000.)
        init_alphas[self.tag_to_ix[START_TAG]] = 0.

        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[0]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            t_r1_k = torch.unsqueeze(feats[feat_index], 0).transpose(0, 1)  # +1
            aa = gamar_r_l + t_r1_k + self.transitions
            forward_var_list.append(torch.logsumexp(aa, dim=1))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)[0]
        return alpha

    def _forward_alg_new_parallel(self, feats):
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -100000000.).cuda(1)
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]].repeat([feats.shape[0], 1])
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).unsqueeze(dim=0)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_out = lstm_out.squeeze()
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _get_lstm_features_parallel(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags.view(-1)])

        for i, feat in enumerate(feats):
            score = score + \
                    self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _score_sentence_parallel(self, feats, tags):

        score = torch.zeros(tags.shape[0]).cuda(1)
        tags = torch.cat([torch.full([tags.shape[0], 1], self.tag_to_ix[START_TAG]).long(), tags], dim=1)
        for i in range(feats.shape[1]):
            feat = feats[:, i, :]
            score = score + \
                    self.transitions[tags[:, i + 1], tags[:, i]] + feat[range(feat.shape[0]), tags[:, i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[:, -1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -100000000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []  
            viterbivars_t = [] 

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG] 
        best_path.reverse()
        return path_score, best_path

    def _viterbi_decode_new(self, feats):
        backpointers = []

        init_vvars = torch.full((1, self.tagset_size), -100000000.).cuda(1)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var_list = []
        forward_var_list.append(init_vvars)

        for feat_index in range(feats.shape[0]):
            if len(feats.shape) == 2:
                gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[1])
            else:
                gamar_r_l = torch.stack([forward_var_list[feat_index]] * 1)
            gamar_r_l = torch.squeeze(gamar_r_l)
            next_tag_var = gamar_r_l + self.transitions
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=1)

            t_r1_k = torch.unsqueeze(feats[feat_index], 0)
            forward_var_new = torch.unsqueeze(viterbivars_t, 0) + t_r1_k

            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t.tolist())

        terminal_var = forward_var_list[-1] + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = torch.argmax(terminal_var).tolist()
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        start = best_path.pop()
        if start != self.tag_to_ix[START_TAG]:
            print('Sanity check error')
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg_new(feats)
        gold_score = self._score_sentence(feats, tags)[0]
        return forward_score - gold_score

    def neg_log_likelihood_parallel(self, sentences, tags):
        feats = self._get_lstm_features_parallel(sentences)
        forward_score = self._forward_alg_new_parallel(feats)
        gold_score = self._score_sentence_parallel(feats, tags.cpu())
        return torch.sum(forward_score - gold_score)

    def forward(self, sentence): 
        lstm_feats = self._get_lstm_features(sentence)

        score, tag_seq = self._viterbi_decode_new(lstm_feats)
        return score, tag_seq


def test(model, test_batch, word_to_ix, tag_to_ix):
    all = 0
    acc = 0
    cnt = 0
    for batch in test_batch:
        if cnt % 100 == 0:
            print('test', cnt)
        cnt = cnt + 1
        for sent, tag in batch:
            sent_tmp = sent.copy()
            sent_tmp.append('<PAD>')
            precheck_sent = prepare_sequence(sent_tmp, word_to_ix)
            for i, t in enumerate(model(precheck_sent)[1][:-1]):
                all = all + 1
                if tag[i] in tag_to_ix:
                    if t == tag_to_ix[tag[i]]:
                        acc = acc + 1
    return acc / all


if __name__ == '__main__':
    START_TAG = "<START>"
    STOP_TAG = "<STOP>"
    PAD_TAG = "<PAD>"
    OUT_TAG = "<OUT>"
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 256
    batch_num = 5724  # 2862
    total_batch = []
    total_sent = []
    for i in range(2862):
        tmp = joblib.load('./dataset/pos/data_set' + str(i) + '.pkl')
        for t in tmp:
            total_sent.append(t)
    random.shuffle(total_sent)
    for i in range(batch_num):
        total_batch.append(total_sent[i*50:i*50+50])
    train_len = int(0.7 * len(total_batch))
    valid_len = int(0.1 * len(total_batch))
    train_batch = total_batch[:train_len]
    valid_batch = total_batch[train_len:train_len + valid_len]
    test_batch = total_batch[train_len + valid_len:]
    training_data = [d for b in train_batch for d in b]
    word_to_ix = {}
    tag_to_ix = {}
    ix_to_tag = {}
    word_to_ix['<PAD>'] = 0
    word_to_ix['<OUT>'] = 1
    for sentence, tags in training_data:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for t in tags:
            if t not in tag_to_ix:
                tag_to_ix[t] = len(tag_to_ix)
                ix_to_tag[tag_to_ix[t]] = t
    for t in [START_TAG, STOP_TAG, PAD_TAG, OUT_TAG]:
        tag_to_ix[t] = len(tag_to_ix)
        ix_to_tag[tag_to_ix[t]] = t
    print(tag_to_ix)

    model = BiLSTM_CRF_MODIFY_PARALLEL(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM).cuda(1)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    best_valid = 0
    for epoch in range(30):  
        cnt = 0
        print('epoch', epoch)
        for training_data in train_batch:
            if cnt % 100 == 0:
                print('begin', cnt)
            cnt = cnt + 1
            model.zero_grad()
            sentence_in_pad, targets_pad = prepare_sequence_batch(training_data, word_to_ix, tag_to_ix)
            loss = model.neg_log_likelihood_parallel(sentence_in_pad, targets_pad)
            loss.backward()
            optimizer.step()
        test_acc = test(model, test_batch, word_to_ix, tag_to_ix)
        valid_acc = test(model, valid_batch, word_to_ix, tag_to_ix)
        if valid_acc > best_valid:
            best_valid = valid_acc
            torch.save(model, './bestModel/POS_Epoch' + str(epoch) + 'acc' + str(test_acc) + '.pt')
        print('epoch:', epoch, 'valid_acc:', valid_acc, 'test_acc', test_acc)
