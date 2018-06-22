# coding:utf-8
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from hparams import hparams


class BahdanauAttention(nn.Module):
    def __init__(self, dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, query, processed_memory):
        """
        Args:
            query: (batch, 1, dim) or (batch, dim)
            processed_memory: (batch, max_time, dim)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        # (batch, 1, dim)

        processed_query = self.query_layer(query)

        # (batch, max_time, 1)
        alignment = self.v(self.tanh(processed_query + processed_memory))

        # (batch, max_time)
        return alignment.squeeze(-1)


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length

    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return ~mask


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=hparams.num_heads,
                 is_masked=False,
                 _style='conv'):
        super(MultiHeadAttention, self).__init__()

        # if query_dim != key_dim:
        #     raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(data=[key_dim], requires_grad=True, dtype=torch.float32)
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False) if _style == 'linear' else \
            nn.Conv1d(1, num_units, 1)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False) if _style == 'linear' else \
            nn.Conv1d(hparams.style_token, num_units, 1)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False) if _style == 'linear' else \
            nn.Conv1d(hparams.style_token, num_units, 1)
        self.bn = nn.BatchNorm1d(num_units)

    def forward(self, query, keys):
        Q = self.query_layer(query)  # [B,L,Dq]
        K = self.key_layer(keys)  # [B,L1,Dk]
        V = self.value_layer(keys)  # [B,L1,Dk]

        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=1)
        K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=1)
        V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=1)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self._key_dim).cuda()
        # use masking (usually for decoder) to prevent leftward
        # information flow and retains auto-regressive property
        # as said in the paper
        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril()
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())
            mask = torch.ones(diag_mat.size()) * (-2 ** 32 + 1)
            # this is some trick that I use to combine the lower diagonal
            # matrix and its masking. (diag_mat-1).abs() will reverse the value
            # inside diag_mat, from 0 to 1 and 1 to zero. with this
            # we don't need loop operation andn could perform our calculation
            # faster
            attention = (attention * diag_mat) + (mask * (diag_mat - 1).abs())
        # put it to softmax
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        # convert attention back to its input original size
        restore_chunk_size = int(attention.size(0) / self._h)
        attention = torch.cat(attention.split(split_size=restore_chunk_size, dim=1), dim=2)
        # residual connection
        attention += query
        # apply batch normalization
        attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)

        return attention


class MultiHeadAttentionTest(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.5,
                 h=hparams.num_heads,
                 is_masked=False,
                 use_dropout=False,
                 _style='conv'):
        super(MultiHeadAttentionTest, self).__init__()

        # if query_dim != key_dim:
        #     raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")
        self.use_dropout = use_dropout
        self._num_units = num_units
        self._h = h
        self._key_dim = torch.tensor(data=[key_dim], requires_grad=True, dtype=torch.float32)
        self._dropout_p = dropout_p
        self._is_masked = is_masked
        self.v = nn.Parameter(torch.randn([num_units]))
        self.use_batchnorm = False
        self.use_residual = False

        self.query_layer = nn.Linear(query_dim, num_units, bias=False) if _style == 'linear' else \
            nn.Conv1d(query_dim, num_units, 1)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False) if _style == 'linear' else \
            nn.Conv1d(key_dim, num_units, 1)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False) if _style == 'linear' else \
            nn.Conv1d(key_dim, num_units, 1)
        self.bn = nn.BatchNorm1d(num_units)

    def __split_last_dim(self, x, heads=None):
        if heads is None:
            heads = self._h
        # return shape [batch, length_x, num_heads, dim_x/num_heads]
        size = x.size()
        new_size = size[:-1] + (heads, int(size[-1] / heads))
        x = x.view(*new_size)
        return x

    def __split_head(self, q, k, v):
        # return [batch,num_heads, length_x, dim_x/num_heads]
        qs = self.__split_last_dim(q).permute(0, 2, 1, 3)
        ks = self.__split_last_dim(k).permute(0, 2, 1, 3)
        vs = self.__split_last_dim(v).permute(0, 2, 1, 3)
        # vs = v.unsqueeze(1).repeat(1, self._h, 1, 1)
        return qs, ks, vs

    def __combine_head(self, x):
        # [batch, length_x,num_heads, dim_x/num_heads]
        x = x.permute(0, 2, 1, 3).contiguous()
        size = x.size()
        new_size = size[:-2] + (size[2] * size[3],)
        return x.view(*new_size)

    def forward(self, query, keys):
        Q = self.query_layer(query.permute(0, 2, 1)).permute(0,2,1)  # [B,L,Dq]
        K = self.key_layer(keys.permute(0, 2, 1)).permute(0,2,1)  # [B,L,Dk]
        V = self.value_layer(keys.permute(0, 2, 1)).permute(0,2,1)  # [B,L,Dk]
        Q, K, V = self.__split_head(Q, K, V)
        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        # chunk_size = int(self._num_units / self._h)
        # Q = torch.cat(Q.split(split_size=chunk_size, dim=2), dim=1)
        # K = torch.cat(K.split(split_size=chunk_size, dim=2), dim=1)
        # V = torch.cat(V.split(split_size=chunk_size, dim=2), dim=1)

        # calculate QK^T
        attention = torch.matmul(Q, K.transpose(2, 3))
        # normalize with sqrt(dk)
        attention = attention / torch.sqrt(self._key_dim).cuda()
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        if self.use_dropout:
            attention = F.dropout(attention, self._dropout_p)
        # multiplyt it with V
        attention = torch.matmul(attention, V)
        attention = self.__combine_head(attention)
        # residual connection
        if self.use_residual:
            attention += query
        # apply batch normalization
        if self.use_dropout:
            attention = self.bn(attention.transpose(1, 2)).transpose(1, 2)

        return attention


class AttentionWrapper(nn.Module):
    def __init__(self, rnn_cell, attention_mechanism,
                 score_mask_value=-float("inf")):
        super(AttentionWrapper, self).__init__()
        self.rnn_cell = rnn_cell
        self.attention_mechanism = attention_mechanism
        self.score_mask_value = score_mask_value

    def forward(self, query, attention, cell_state, memory,
                processed_memory=None, mask=None, memory_lengths=None):
        if processed_memory is None:
            processed_memory = memory
        if memory_lengths is not None and mask is None:
            mask = get_mask_from_lengths(memory, memory_lengths)

        # Concat input query and previous attention context
        cell_input = torch.cat((query, attention), -1)

        # Feed it to RNN
        cell_output = self.rnn_cell(cell_input, cell_state)
        # GRUCell（LSTMCell） 和GRU（LSTM）的区别在哪里？
        # GRUCell处理的是一个输入，利用前面的cell_input 和cell_state，GRU处理的是整个的序列，输出整个序列的output_state和output
        # Alignment
        # (batch, max_time)
        alignment = self.attention_mechanism(query=cell_output, processed_memory=processed_memory)
        # attention
        if mask is not None:
            mask = mask.view(query.size(0), -1)
            alignment.data.masked_fill_(mask, self.score_mask_value)

        # Normalize attention weight
        alignment = F.softmax(alignment)

        # Attention context vector
        # (batch, 1, dim)
        # [B,1,T].dot([B,T,N]) ==>[B,1,N]
        attention = torch.bmm(alignment.unsqueeze(1), memory)

        # (batch, dim)
        attention = attention.squeeze(1)

        return cell_output, attention, alignment
