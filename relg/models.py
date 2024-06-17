from torch import nn
import torch
import torch.nn.functional as F

import numpy as np

from .utils import encode_onehot
from .layers import MLP
from .activations import gumbel_softmax


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        rnn_hidden_dim,
        num_objects,
        num_edge_types,
        use_bn=True,
        dropout=0,
        **kwargs,
    ):
        super().__init__()
        self.num_objects = num_objects
        self.num_edges = num_edge_types
        self.set_edges(self.num_objects)

        self.mlp1 = MLP(input_dim, hidden_dim, hidden_dim, dropout, no_bn=use_bn)
        self.mlp2 = MLP(2 * hidden_dim, hidden_dim, hidden_dim, dropout, no_bn=use_bn)
        self.mlp3 = MLP(hidden_dim, hidden_dim, hidden_dim, dropout, no_bn=use_bn)
        self.mlp4 = MLP(3 * hidden_dim, hidden_dim, hidden_dim, dropout, no_bn=use_bn)

        self.forward_rnn = nn.LSTM(hidden_dim, rnn_hidden_dim, batch_first=True)
        self.reverse_rnn = nn.LSTM(hidden_dim, rnn_hidden_dim, batch_first=True)

        self.mlp_out = nn.Linear(2 * rnn_hidden_dim, self.num_edges)
        self.mlp_prior = nn.Linear(rnn_hidden_dim, self.num_edges)

    def set_edges(self, num_objects):
        edges = np.ones(num_objects) - np.eye(num_objects)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.edge2node_mat = nn.Parameter(
            torch.FloatTensor(
                encode_onehot(self.recv_edges).transpose(),
            ),
            requires_grad=False,
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def node2edge(self, node_embeddings):
        # [batch, num_objects, timesteps, embed_size]
        send_embed = node_embeddings[:, self.send_edges, :, :]
        recv_embed = node_embeddings[:, self.recv_edges, :, :]
        return torch.cat([send_embed, recv_embed], dim=-1)

    def edge2node(self, edge_embeddings):
        # [batch, num_edges, timesteps, embed_size]
        old_shape = edge_embeddings.shape
        tmp_embeddings = edge_embeddings.view(old_shape[0], old_shape[1], -1)
        incoming = torch.matmul(self.edge2node_mat, tmp_embeddings).view(
            old_shape[0], -1, old_shape[2], old_shape[3]
        )
        return incoming / (self.num_objects - 1)

    def forward(self, inputs):
        # inputs: [batch, timesteps, num_objects, input_size]
        num_timesteps = inputs.size(1)
        x = inputs.transpose(2, 1).contiguous()
        x = self.mlp1(x)
        x = self.node2edge(x)
        x = self.mlp2(x)
        x_skip = x
        x = self.edge2node(x)
        x = self.mlp3(x)
        x = self.node2edge(x)
        # Skip connection
        x = torch.cat((x, x_skip), dim=-1)
        x = self.mlp4(x)

        # x: [batch, num_edges, timesteps, hidden_dim]
        shape = x.shape
        x = x.contiguous().view(-1, shape[2], shape[3])
        forward_x, prior_state = self.forward_rnn(x)

        reverse_x = x.flip(1)
        reverse_x, _ = self.reverse_rnn(reverse_x)
        reverse_x = reverse_x.flip(1)

        # x: [batch*num_edges, timesteps, hidden_dim]
        prior = (
            self.mlp_prior(forward_x)
            .view(shape[0], shape[1], shape[2], self.num_edges)
            .transpose(1, 2)
            .contiguous()
        )
        combined = torch.cat([forward_x, reverse_x], dim=-1)
        encoded = (
            self.mlp_out(combined)
            .view(shape[0], shape[1], shape[2], self.num_edges)
            .transpose(1, 2)
            .contiguous()
        )

        return prior, encoded, prior_state


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_edge_types,
        num_objects,
        dropout=0,
        skip_first_edge_type=True,
        **kwargs,
    ):
        super().__init__()

        self.num_objects = num_objects
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.skip_first_edge_type = skip_first_edge_type

        self.num_edges = num_edge_types
        self.set_edges(self.num_objects)

        self.mlps1 = nn.ModuleList(
            [nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_edge_types)]
        )
        self.mlps2 = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_edge_types)]
        )

        # Recurrent interaction decoder
        self.hidden_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.hidden_i = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.hidden_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.input_r = nn.Linear(input_dim, hidden_dim, bias=True)
        self.input_i = nn.Linear(input_dim, hidden_dim, bias=True)
        self.input_n = nn.Linear(input_dim, hidden_dim, bias=True)

        self.mlp_out1 = nn.Linear(hidden_dim, hidden_dim)
        self.mlp_out2 = nn.Linear(hidden_dim, hidden_dim)

        self.mlp_out3 = nn.Linear(hidden_dim, input_dim)

    def set_edges(self, num_objects):
        edges = np.ones(num_objects) - np.eye(num_objects)
        self.send_edges = np.where(edges)[0]
        self.recv_edges = np.where(edges)[1]

        self.edge2node_mat = nn.Parameter(
            torch.FloatTensor(
                encode_onehot(self.recv_edges).transpose(),
            ),
            requires_grad=False,
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs, hidden, edges):
        # inputs: [batch, num_objects, input_size]
        # hidden: [batch, num_objects, rnn_hidden_dim]
        # edgesf: [batch, num_edges, num_edge_types]
        if self.training:
            dropout_prob = self.dropout
        else:
            dropout_prob = 0.0

        # node2edge
        receivers = hidden[:, self.recv_edges, :]
        senders = hidden[:, self.send_edges, :]

        # pre_msg: [batch, num_edges, 2*rnn_hidden_dim]
        pre_msg = torch.cat([receivers, senders], dim=-1)

        all_msgs = torch.zeros(
            pre_msg.size(0), pre_msg.size(1), self.hidden_dim, device=inputs.device
        )

        if self.skip_first_edge_type:
            start_idx = 1
            norm = float(len(self.mlps2)) - 1
        else:
            start_idx = 0
            norm = float(len(self.mlps2))

        # Run separate MLP for every edge type
        for i in range(start_idx, len(self.mlps2)):
            msg = torch.tanh(self.mlps1[i](pre_msg))
            msg = F.dropout(msg, p=dropout_prob)
            msg = torch.tanh(self.mlps2[i](msg))
            msg = msg * edges[:, :, i : i + 1]
            all_msgs += msg / norm

        # Message aggregation
        agg_msgs = torch.matmul(self.edge2node_mat, all_msgs)
        agg_msgs = agg_msgs.contiguous() / (self.num_objects - 1)

        # GRU-style gated aggregation
        inp_r = self.input_r(inputs).view(inputs.size(0), self.num_objects, -1)
        inp_i = self.input_i(inputs).view(inputs.size(0), self.num_objects, -1)
        inp_n = self.input_n(inputs).view(inputs.size(0), self.num_objects, -1)
        r = torch.sigmoid(inp_r + self.hidden_r(agg_msgs))
        i = torch.sigmoid(inp_i + self.hidden_i(agg_msgs))
        n = torch.tanh(inp_n + r * self.hidden_h(agg_msgs))
        hidden = (1 - i) * n + i * hidden

        # Output MLP
        pred = F.dropout(F.relu(self.mlp_out1(hidden)), p=dropout_prob)
        pred = F.dropout(F.relu(self.mlp_out2(pred)), p=dropout_prob)

        pred = self.mlp_out3(pred)
        pred = inputs + pred

        return pred, hidden







# class dnri(nn.Module):
#     def __init__(
#         self,
#         input_dim,
#         encoder_hidden_dim,
#         rnn_encoder_dim,
#         decoder_dim,
#         num_objects,
#         num_edge_types,
#         kl_weight=1.0,
#         gumbel_temp=0.5,
#         prior=None,
#         **kwargs,
#     ):
#         super().__init__()
#         self.encoder = Encoder(
#             input_dim=input_dim,
#             hidden_dim=encoder_hidden_dim,
#             rnn_hidden_dim=rnn_encoder_dim,
#             num_objects=num_objects,
#             num_edge_types=num_edge_types,
#         ).cpu()
#         self.decoder = Decoder(
#             input_dim=input_dim,
#             hidden_dim=decoder_dim,
#             num_objects=num_objects,
#             num_edge_types=num_edge_types,
#         ).cpu()

#         self.kl_weight = kl_weight
#         self.gumbel_temp = gumbel_temp

#         self.set_prior(prior)

#     def set_prior(self, prior):
#         if prior:
#             self.prior = prior
#             self.log_prior = torch.FloatTensor(np.log(prior)).cpu()
#         else:
#             prior = np.zeros(self.encoder.num_edges)
#             prior.fill(1.0 / self.encoder.num_edges)
#             self.log_prior = torch.FloatTensor(np.log(prior)).cpu()

#     def forward(self, inputs):
#         hidden = self.get_initial_hidden(inputs)
#         timesteps = inputs.size(1)

#         edges_p, edges_enc = [], []

#         for step in range(timesteps - 1):
#             dinput = inputs[:, step]
#             prior_logits, posterior_logits, _ = self.encoder(dinput)
#             prediction, hidden, edge = self.step_forward(
#                 dinput,
#                 hidden,
#                 posterior_logits,
#                 hard_sample=not self.training,
#             )
#             edges_p.append(prior_logits)
#             edges_enc.append(posterior_logits)
        
#         edges_p = torch.stack(edges_p, dim=1)
#         edges_enc = torch.stack(edges_enc, dim=1)

#         return edges_p, edges_enc, hidden

#     def get_initial_hidden(self, inputs):
#         return torch.zeros(
#             inputs.size(0),
#             inputs.size(2),
#             self.decoder.hidden_dim,
#             device=inputs.device,
#         ).cpu()

#     def step_forward(
#         self,
#         inputs,
#         decoder_hidden,
#         edge_logits,
#         hard_sample,
#         **kwargs,
#     ):
#         old_shape = edge_logits.shape
#         edges = gumbel_softmax(
#             edge_logits.reshape(-1, self.encoder.num_edges),
#             tau=self.gumbel_temp,
#             hard=hard_sample,
#         ).view(old_shape)

#         predictions, decoder_hidden = self.decoder(
#             inputs,
#             decoder_hidden,
#             edges,
#         )
#         return predictions, decoder_hidden, edges














class dnri(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_hidden_dim,
        rnn_encoder_dim,
        decoder_dim,
        num_objects,
        num_edge_types,
        kl_weight=1.0,
        gumbel_temp=0.5,
        prior=None,
        **kwargs,
    ):
        super().__init__()
        self.encoder = Encoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            rnn_hidden_dim=rnn_encoder_dim,
            num_objects=num_objects,
            num_edge_types=num_edge_types,
        ).cpu()
        self.decoder = Decoder(
            input_dim=input_dim,
            hidden_dim=decoder_dim,
            num_objects=num_objects,
            num_edge_types=num_edge_types,
        ).cpu()

        self.kl_weight = kl_weight
        self.gumbel_temp = gumbel_temp

        self.set_prior(prior)

    def get_initial_hidden(self, inputs):
        return torch.zeros(
            inputs.size(0),
            inputs.size(2),
            self.decoder.hidden_dim,
            device=inputs.device,
        ).cpu()

    def set_prior(self, prior):
        if prior:
            self.prior = prior
            self.log_prior = torch.FloatTensor(np.log(prior)).cpu()
        else:
            prior = np.zeros(self.encoder.num_edges)
            prior.fill(1.0 / self.encoder.num_edges)
            self.log_prior = torch.FloatTensor(np.log(prior)).cpu()

    def nll_gaussian(self, preds, target, variance=5e-5):
        neg_log_p = (preds - target) ** 2 / (2 * variance)
        const = 0
        return (neg_log_p.sum(-1) + const).view(preds.size(0), -1).mean(dim=1)

    def kl_categorical_learned(self, preds, prior_logits):
        log_prior = nn.LogSoftmax(dim=-1)(prior_logits)
        kl_div = preds * (torch.log(preds + 1e-16) - log_prior)
        return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)

    def kl_categorical_avg(self, preds, eps=1e-16):
        avg_preds = preds.mean(dim=2)
        kl_div = avg_preds * (torch.log(avg_preds + eps) - self.log_prior)
        return kl_div.sum(-1).view(preds.size(0), -1).mean(dim=1)

    def step_forward(
        self,
        inputs,
        decoder_hidden,
        edge_logits,
        hard_sample,
        **kwargs,
    ):
        old_shape = edge_logits.shape
        edges = gumbel_softmax(
            edge_logits.reshape(-1, self.encoder.num_edges),
            tau=self.gumbel_temp,
            hard=hard_sample,
        ).view(old_shape)

        predictions, decoder_hidden = self.decoder(
            inputs,
            decoder_hidden,
            edges,
        )
        return predictions, decoder_hidden, edges

    def compute_loss(self, inputs):
        hidden = self.get_initial_hidden(inputs)
        timesteps = inputs.size(1)

        edges, predictions = [], []

        sinputs = inputs[:, :-1]
        prior_logits, posterior_logits, _ = self.encoder(sinputs)

        for step in range(timesteps - 1):
            dinput = inputs[:, step]
            prediction, hidden, edge = self.step_forward(
                dinput,
                hidden,
                posterior_logits[:, step],
                hard_sample=not self.training,
            )
            predictions.append(prediction)
            edges.append(edge)

        predictions = torch.stack(predictions, dim=1)
        target = inputs[:, 1:]
        loss_nll = self.nll_gaussian(predictions, target)

        prob = F.softmax(posterior_logits, dim=-1)
        loss_kl = self.kl_categorical_learned(prob, prior_logits)

        # add uniform prior
        loss_kl = 0.5 * loss_kl + 0.5 * self.kl_categorical_avg(prob)

        loss = loss_nll + self.kl_weight * loss_kl
        loss = loss.mean()

        return loss, loss_nll, loss_kl
