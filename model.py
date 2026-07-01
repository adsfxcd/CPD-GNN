import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot, zeros

import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import RGCNConv

import torch.fft
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from graph import batch_graphify


# ==================== TeLU Activation Function ====================

class TeLU(nn.Module):
    """TeLU activation function: x * tanh(exp(x))"""
    def forward(self, x):
        return x * torch.tanh(torch.exp(x))


# ==================== Prompt-based Cross-Modal Interaction Modules ====================

class Mlp(nn.Module):
    """Multi-Layer Perceptron with GELU activation"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def calc_cosine_similarity(feat1, feat2):
    """
    Calculate cosine similarity between two feature tensors
    feat1: [batch, m, dim]
    feat2: [batch, n, dim]
    Returns: [batch*m, batch*n] similarity matrix
    """
    batch, m, dim = feat1.size()
    n = feat2.size(1)
    fea1_reshape = feat1.reshape(-1, dim)
    fea2_reshape = feat2.reshape(-1, dim)
    fea1_norm = fea1_reshape / (torch.norm(fea1_reshape, dim=1, keepdim=True) + 1e-8)
    fea2_norm = fea2_reshape / (torch.norm(fea2_reshape, dim=1, keepdim=True) + 1e-8)
    cosine_similarity = torch.mm(fea1_norm, fea2_norm.t())
    return cosine_similarity


class PromptFormer(nn.Module):
    """
    Prompt-based Cross-Modal Interaction Module
    Generates prototypes from input features and creates prompts through similarity matching
    Uses adaptive pooling to handle dynamic sequence lengths
    """
    def __init__(self, seq_len, n_proto, D_e, prompt_dim, dropout=0.1):
        super(PromptFormer, self).__init__()
        self.seq_len = seq_len
        self.n_proto = n_proto
        self.D_e = D_e
        self.prompt_dim = prompt_dim

        # Use adaptive pooling + linear to handle dynamic sequence lengths
        # [batch, D_e, seq_len] -> [batch, D_e, n_proto] via adaptive pooling
        self.adaptive_pool = nn.AdaptiveAvgPool1d(n_proto)
        # Transform pooled features
        self.proto_transform = nn.Sequential(
            nn.Linear(D_e, D_e),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(D_e, D_e)
        )
        # Project D_e to prompt_dim
        self.proj_d = Mlp(in_features=D_e, hidden_features=prompt_dim, out_features=prompt_dim, drop=dropout)
        # Residual projection
        self.proj_prompt = nn.Linear(D_e, prompt_dim)

    def forward(self, x, umask):
        """
        Args:
            x: [batch, seq_len, D_e] - input features (dynamic seq_len)
            umask: [batch, seq_len] - utterance mask (1 for valid, 0 for padding)
        Returns:
            prompt: [batch, seq_len, prompt_dim]
            prototype: [batch, n_proto, D_e]
            res: [batch, seq_len, prompt_dim] - residual connection
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Generate prototypes using adaptive pooling (handles dynamic seq_len)
        # [batch, seq_len, D_e] -> [batch, D_e, seq_len] -> [batch, D_e, n_proto] -> [batch, n_proto, D_e]
        x_transposed = x.permute(0, 2, 1)  # [batch, D_e, seq_len]
        pooled = self.adaptive_pool(x_transposed)  # [batch, D_e, n_proto]
        prototype = pooled.permute(0, 2, 1)  # [batch, n_proto, D_e]
        prototype = self.proto_transform(prototype)  # [batch, n_proto, D_e]

        # Calculate similarity between input and prototypes
        sim = calc_cosine_similarity(x, prototype)  # [batch*seq_len, batch*n_proto]

        # Reshape prototype for attention
        prototype_flat = prototype.reshape(-1, self.D_e)  # [batch*n_proto, D_e]

        # Create mask for softmax
        mask = umask.view(-1).unsqueeze(1).repeat(1, self.n_proto * batch_size)  # [batch*seq_len, batch*n_proto]
        mask = (1 - mask.float()) * -10000.0
        sim = sim + mask
        sim = F.softmax(sim, dim=1)

        # Generate prompt through weighted sum of prototypes
        prompt = sim @ prototype_flat  # [batch*seq_len, D_e]
        prompt = prompt.reshape(batch_size, seq_len, self.D_e)  # [batch, seq_len, D_e]

        # Project to prompt dimension
        prompt = self.proj_d(prompt)  # [batch, seq_len, prompt_dim]
        res = self.proj_prompt(x)  # [batch, seq_len, prompt_dim]
        prompt = prompt + res  # Residual connection

        return prompt, prototype, res


class PromptFormer_v2(nn.Module):
    """
    Second stage PromptFormer that uses pre-computed prototypes
    No need for proj_n as it receives pre-computed prototypes
    """
    def __init__(self, seq_len, n_proto, D_e, prompt_dim, dropout=0.1):
        super(PromptFormer_v2, self).__init__()
        self.seq_len = seq_len
        self.n_proto = n_proto
        self.D_e = D_e
        self.prompt_dim = prompt_dim

        # Only need projection layers, not prototype generation
        self.proj_d = Mlp(in_features=D_e, hidden_features=prompt_dim, out_features=prompt_dim, drop=dropout)
        self.proj_prompt = nn.Linear(D_e, prompt_dim)

    def forward(self, x, umask, prototype):
        """
        Args:
            x: [batch, seq_len, D_e] - dynamic seq_len
            umask: [batch, seq_len]
            prototype: [batch, n_proto, D_e] - pre-computed prototypes
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        sim = calc_cosine_similarity(x, prototype)
        prototype_flat = prototype.reshape(-1, self.D_e)

        mask = umask.view(-1).unsqueeze(1).repeat(1, self.n_proto * batch_size)
        mask = (1 - mask.float()) * -10000.0  # Fixed: added .float()
        sim = sim + mask
        sim = F.softmax(sim, dim=1)

        prompt = sim @ prototype_flat
        prompt = prompt.reshape(batch_size, seq_len, self.D_e)

        prompt = self.proj_d(prompt)
        res = self.proj_prompt(x)
        prompt = prompt + res

        return prompt


class CrossModalPromptFusion(nn.Module):
    """
    Cross-Modal Prompt Fusion Module
    Enables information exchange between different modalities through prompts
    """
    def __init__(self, adim, tdim, vdim, D_e, prompt_dim=16, n_proto=10, dropout=0.1, max_seq_len=150):
        super(CrossModalPromptFusion, self).__init__()
        self.D_e = D_e
        self.prompt_dim = prompt_dim
        self.n_proto = n_proto

        # Modal-specific input projections
        self.a_in_proj = nn.Sequential(nn.Linear(adim, D_e), nn.Dropout(dropout))
        self.t_in_proj = nn.Sequential(nn.Linear(tdim, D_e), nn.Dropout(dropout))
        self.v_in_proj = nn.Sequential(nn.Linear(vdim, D_e), nn.Dropout(dropout))

        # Shared-Private encoders for each modality
        self.shared_enc_a = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=dropout)
        self.shared_enc_t = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=dropout)
        self.shared_enc_v = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=dropout)

        self.private_enc_a = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=dropout)
        self.private_enc_t = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=dropout)
        self.private_enc_v = Mlp(in_features=D_e, hidden_features=D_e, out_features=D_e, drop=dropout)

        # Decoders for reconstruction
        self.dec_a = Mlp(in_features=D_e*2, hidden_features=D_e, out_features=D_e, drop=dropout)
        self.dec_t = Mlp(in_features=D_e*2, hidden_features=D_e, out_features=D_e, drop=dropout)
        self.dec_v = Mlp(in_features=D_e*2, hidden_features=D_e, out_features=D_e, drop=dropout)

        # PromptFormers for cross-modal interaction
        self.prompt_former_a = PromptFormer(max_seq_len, n_proto, D_e, prompt_dim, dropout)
        self.prompt_former_t = PromptFormer(max_seq_len, n_proto, D_e, prompt_dim, dropout)
        self.prompt_former_v = PromptFormer(max_seq_len, n_proto, D_e, prompt_dim, dropout)

        # Cross-modal fusion projections (fuse prompts from other modalities)
        self.fusion_proj_a = nn.Linear(D_e + 2 * prompt_dim, D_e)
        self.fusion_proj_t = nn.Linear(D_e + 2 * prompt_dim, D_e)
        self.fusion_proj_v = nn.Linear(D_e + 2 * prompt_dim, D_e)

        # Router for dynamic modality weighting
        self.router = Mlp(in_features=D_e * 3, hidden_features=D_e, out_features=3, drop=dropout)

        # Output projection
        self.output_proj = nn.Linear(D_e * 3, adim + tdim + vdim)

    def forward(self, audio, text, video, umask, input_mask=None):
        """
        Args:
            audio: [seq_len, batch, adim]
            text: [seq_len, batch, tdim]
            video: [seq_len, batch, vdim]
            umask: [batch, seq_len] - utterance mask
            input_mask: [seq_len, batch, 3] - modality availability mask (optional)
        Returns:
            fused_features: [seq_len, batch, adim+tdim+vdim]
            shared_features: (shared_a, shared_t, shared_v) for disentangle loss
            private_features: (private_a, private_t, private_v) for disentangle loss
        """
        seq_len, batch, _ = audio.size()

        # Transpose to [batch, seq_len, dim] for processing
        audio = audio.permute(1, 0, 2)  # [batch, seq_len, adim]
        text = text.permute(1, 0, 2)    # [batch, seq_len, tdim]
        video = video.permute(1, 0, 2)  # [batch, seq_len, vdim]

        # Project to unified dimension D_e
        proj_a = self.a_in_proj(audio)  # [batch, seq_len, D_e]
        proj_t = self.t_in_proj(text)   # [batch, seq_len, D_e]
        proj_v = self.v_in_proj(video)  # [batch, seq_len, D_e]

        # Shared-Private feature separation
        shared_a = self.shared_enc_a(proj_a)    # [batch, seq_len, D_e]
        shared_t = self.shared_enc_t(proj_t)
        shared_v = self.shared_enc_v(proj_v)

        private_a = self.private_enc_a(proj_a)  # [batch, seq_len, D_e]
        private_t = self.private_enc_t(proj_t)
        private_v = self.private_enc_v(proj_v)

        # Generate cross-modal prompts
        prompt_a, proto_a, _ = self.prompt_former_a(shared_a, umask)  # [batch, seq_len, prompt_dim]
        prompt_t, proto_t, _ = self.prompt_former_t(shared_t, umask)
        prompt_v, proto_v, _ = self.prompt_former_v(shared_v, umask)

        # Cross-modal fusion: each modality receives prompts from other modalities
        # Audio receives prompts from Text and Video
        enhanced_a = self.fusion_proj_a(torch.cat([shared_a, prompt_t, prompt_v], dim=-1))
        # Text receives prompts from Audio and Video
        enhanced_t = self.fusion_proj_t(torch.cat([shared_t, prompt_a, prompt_v], dim=-1))
        # Video receives prompts from Audio and Text
        enhanced_v = self.fusion_proj_v(torch.cat([shared_v, prompt_a, prompt_t], dim=-1))

        # Add back private features
        final_a = enhanced_a + private_a  # [batch, seq_len, D_e]
        final_t = enhanced_t + private_t
        final_v = enhanced_v + private_v

        # Router-based dynamic weighting
        x_concat = torch.cat([final_a, final_t, final_v], dim=-1)  # [batch, seq_len, 3*D_e]
        weights = F.softmax(self.router(x_concat), dim=-1)  # [batch, seq_len, 3]
        weights = weights.unsqueeze(-1)  # [batch, seq_len, 3, 1]

        x_stacked = torch.stack([final_a, final_t, final_v], dim=2)  # [batch, seq_len, 3, D_e]
        x_weighted = weights * x_stacked  # [batch, seq_len, 3, D_e]

        # Final fusion
        fused = torch.cat([x_weighted[:, :, 0, :], x_weighted[:, :, 1, :], x_weighted[:, :, 2, :]], dim=-1)
        # [batch, seq_len, 3*D_e]

        # Project back to original dimension
        fused_features = self.output_proj(fused)  # [batch, seq_len, adim+tdim+vdim]

        # Transpose back to [seq_len, batch, dim]
        fused_features = fused_features.permute(1, 0, 2)  # [seq_len, batch, adim+tdim+vdim]

        # Return shared and private features for disentangle loss (transpose to [seq_len, batch, D_e])
        shared_features = (shared_a.permute(1, 0, 2), shared_t.permute(1, 0, 2), shared_v.permute(1, 0, 2))
        private_features = (private_a.permute(1, 0, 2), private_t.permute(1, 0, 2), private_v.permute(1, 0, 2))

        return fused_features, shared_features, private_features


# ==================== End of Prompt-based Modules ====================


class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())  # [batch, seq_len]

        if self.att_type == 'dot':
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)  # [batch, mem_dim, seqlen]
            x_ = self.transform(x).unsqueeze(1)  # [batch, 1, mem_dim]
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)  # [batch, mem_dim, seq_len]
            M_ = M_ * mask_  # [batch, mem_dim, seqlen]
            alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)  # attention value: [batch, 1, seqlen]
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)  # [batch, 1, seqlen]
            alpha_masked = alpha_ * mask.unsqueeze(1)  # [batch, 1, seqlen]
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # [batch, 1, 1]
            alpha = alpha_masked / alpha_sum  # normalized attention: [batch, 1, seqlen]
            # alpha = torch.where(alpha.isnan(), alpha_masked, alpha)
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # [batch, 1, seqlen]

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # [batch, mem_dim]
        return attn_pool, alpha


# change [num_utterance, dim] => [seqlen, batch, dim]
def utterance_to_conversation(outputs, seq_lengths, umask, no_cuda):
    input_conversation_length = torch.tensor(seq_lengths)  # [6, 24, 13, 9]
    start_zero = input_conversation_length.data.new(1).zero_()  # [0]

    if not no_cuda:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)  # [int]
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)  # [0,  6, 30, 43]

    outputs = torch.stack([pad(outputs.narrow(0, s, l), max_len, no_cuda)  # [seqlen, batch, dim]
                           for s, l in zip(start.data.tolist(),
                                           input_conversation_length.data.tolist())], 0).transpose(0, 1)
    return outputs


def pad(tensor, length, no_cuda):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if not no_cuda:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if not no_cuda:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor


def com_mult(a, b):
    r1, i1 = a[..., 0], a[..., 1]
    r2, i2 = b[..., 0], b[..., 1]
    return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)


def conj(a):
    a[..., 1] = -a[..., 1]
    return a


def ccorr(a, b):
    return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))


class STEFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class HypergraphConv(MessagePassing):
    """Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        use_attention (bool, optional): If set to :obj:`True`, attention
            will be added to this layer. (default: :obj:`False`)
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, use_attention=False, heads=1,
                 concat=True, negative_slope=0.2, dropout=0, bias=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(HypergraphConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention
        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(
                torch.Tensor(in_channels, heads * out_channels))
            self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))
        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.edgeweight = Parameter(torch.Tensor(in_channels, out_channels))
        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.edgefc = torch.nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.edgeweight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def forward(self, x, hyperedge_index,
                hyperedge_weight=None,
                hyperedge_attr=None, EW_weight=None, dia_len=None):
        r"""
        Args:
            x (Tensor): Node feature matrix
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            hyperedge_index (LongTensor): The hyperedge indices, *i.e.*
                the sparse incidence matrix
                :math:`\mathbf{H} \in {\{ 0, 1 \}}^{N \times M}` mapping from
                nodes to edges.
            hyperedge_weight (Tensor, optional): Hyperedge weights
                :math:`\mathbf{W} \in \mathbb{R}^M`. (default: :obj:`None`)
            hyperedge_attr (Tensor, optional): Hyperedge feature matrix in
                :math:`\mathbb{R}^{M \times F}`.
                These features only need to get passed in case
                :obj:`use_attention=True`. (default: :obj:`None`)
        """
        num_nodes, num_edges = x.size(0), 0
        if hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1
        if hyperedge_weight is None:
            hyperedge_weight = x.new_ones(num_edges)

        alpha = None
        if self.use_attention:
            assert hyperedge_attr is not None
            x = x.view(-1, self.heads, self.out_channels)
            hyperedge_attr = torch.matmul(hyperedge_attr, self.weight)
            hyperedge_attr = hyperedge_attr.view(-1, self.heads,
                                                 self.out_channels)
            x_i = x[hyperedge_index[0]]
            x_j = hyperedge_attr[hyperedge_index[1]]
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
            alpha = F.leaky_relu(alpha, self.negative_slope)
            alpha = softmax(alpha, hyperedge_index[0], num_nodes=x.size(0))
            alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        D = scatter_add(hyperedge_weight[hyperedge_index[1]],
                        hyperedge_index[0], dim=0, dim_size=num_nodes)  # [num_nodes]
        D = 1.0 / D
        D[D == float("inf")] = 0
        if EW_weight is None:
            B = scatter_add(x.new_ones(hyperedge_index.size(1)),
                            hyperedge_index[1], dim=0, dim_size=num_edges)  # [num_edges]
        else:
            B = scatter_add(EW_weight[hyperedge_index[0]],
                            hyperedge_index[1], dim=0, dim_size=num_edges)  # [num_edges]
        B = 1.0 / B
        B[B == float("inf")] = 0
        self.flow = 'source_to_target'
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha,
                             size=(num_nodes, num_edges))
        self.flow = 'target_to_source'
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha,
                             size=(num_edges, num_nodes))
        if self.concat is True and out.size(1) == 1:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        return torch.nn.LeakyReLU()(out)  #

    def message(self, x_j, norm_i, alpha):
        H, F = self.heads, self.out_channels

        if x_j.dim() == 2:
            out = norm_i.view(-1, 1, 1) * x_j.view(-1, H, F)
        else:
            out = norm_i.view(-1, 1, 1) * x_j
        if alpha is not None:
            out = alpha.view(-1, self.heads, 1) * out

        return out


    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class highConv(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr='add'):
        super(highConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.gate = torch.nn.Linear(2*in_channels, 1)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        #x = self.lin(x)

        # Step 3-5: Start propagating messages.
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        # x_j e.g.[135090, 512]

        # Step 3: Normalize node features.
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        #h2 = x_i - x_j
        h2 = torch.cat([x_i, x_j], dim=1)
        alpha_g = torch.tanh(self.gate(h2))#e.g.[135090, 1]

        return norm.view(-1, 1) * (x_j) *alpha_g

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        # Step 5: Return new node embeddings.
        return aggr_out



class GraphNetwork(torch.nn.Module):
    def __init__(self, num_features, num_relations, time_attn, hidden_size=64, dropout=0.5, no_cuda=False):

        super(GraphNetwork, self).__init__()
        self.no_cuda = no_cuda
        self.time_attn = time_attn
        self.hidden_size = hidden_size

        ## graph modeling
        self.conv1 = RGCNConv(num_features, hidden_size, num_relations)
        self.hypergraph= HypergraphConv(in_channels=hidden_size, out_channels=hidden_size)  # hypergraph
        self.high_conv = highConv(in_channels=hidden_size, out_channels=hidden_size)  # highConv

        ## nodal attention
        D_h = num_features + hidden_size
        self.grufusion = nn.GRU(input_size=D_h, hidden_size=D_h, num_layers=2, bidirectional=True, dropout=dropout)

        ## sequence attention
        self.matchatt = MatchingAttention(2 * D_h, 2 * D_h, att_type='general2')
        self.linear = nn.Linear(2 * D_h, D_h)

        ## TeLU activation
        self.telu = TeLU()

    def forward(self, features, edge_index, edge_type, seq_lengths, umask):
        '''
        features: input node features: [num_nodes, in_channels]
        edge_index: [2, edge_num]
        edge_type: [edge_num]
        '''


        ## graph model: graph => outputs
        out = self.conv1(features, edge_index, edge_type)  # [num_features -> hidden_size]

        out = self.hypergraph(out, edge_index) # 浣跨敤hypergraph灞?  [hidden_size -> hidden_size]

        out = self.high_conv(out, edge_index)  # 浣跨敤highConv灞?  [hidden_size -> hidden_size]

        outputs = torch.cat([features, out], dim=-1)  # [num_nodes, num_features(16)+hidden_size(8)]

        ## change utterance to conversation: (outputs->outputs)
        outputs = outputs.reshape(-1, outputs.size(1))  # [num_utterance, dim]
        outputs = utterance_to_conversation(outputs, seq_lengths, umask, self.no_cuda)  # [seqlen, batch, dim]
        outputs = outputs.reshape(outputs.size(0), outputs.size(1), 1, -1)  # [seqlen, batch, ?, dim]

        ## outputs -> outputs:
        seqlen = outputs.size(0)
        batch = outputs.size(1)
        outputs = torch.reshape(outputs, (seqlen, batch, -1))  # [seqlen, batch, dim]
        outputs = self.grufusion(outputs)[0]  # [seqlen, batch, dim]

        ## outputs -> hidden:
        ## sequence attention => [seqlen, batch, d_h]
        if self.time_attn:
            alpha = []
            att_emotions = []
            for t in outputs:  # [bacth, dim]
                # att_em: [batch, mem_dim] # alpha_: [batch, 1, seqlen]
                att_em, alpha_ = self.matchatt(outputs, t, mask=umask)
                att_emotions.append(att_em.unsqueeze(0))  # [1, batch, mem_dim]
                alpha.append(alpha_[:, 0, :])  # [batch, seqlen]
            att_emotions = torch.cat(att_emotions, dim=0)  # [seqlen, batch, mem_dim]
            hidden = self.telu(self.linear(att_emotions))  # [seqlen, batch, D_h]
        else:
            alpha = []
            hidden = self.telu(self.linear(outputs))  # [seqlen, batch, D_h]

        return hidden,


'''
base_model: LSTM or GRU
adim, tdim, vdim: input feature dim
D_e: hidder feature dimensions of base_model is 2*D_e
D_g, D_p, D_h, D_a, graph_hidden_size
'''


class GraphModel(nn.Module):
    """Graph model without prompt-based fusion."""
    def __init__(self, base_model, adim, tdim, vdim, D_e, graph_hidden_size,n_speakers, window_past, window_future,
                 n_classes, dropout=0.5, time_attn=True, no_cuda=False,use_bn=False):
        super(GraphModel, self).__init__()
        self.no_cuda = no_cuda
        self.base_model = base_model
        # Change input features => 2*D_e
        self.lstm = nn.LSTM(input_size=adim + tdim + vdim, hidden_size=D_e, num_layers=2, bidirectional=True,dropout=dropout)
        self.gru = nn.GRU(input_size=adim + tdim + vdim, hidden_size=D_e, num_layers=2, bidirectional=True,dropout=dropout)
        self.n_speakers = n_speakers
        self.window_past = window_past
        self.window_future = window_future
        self.time_attn = time_attn
        ## gain graph models for 'temporal' and 'speaker'
        n_relations = 3
        self.graph_net_temporal = GraphNetwork(2 * D_e, n_relations, self.time_attn, graph_hidden_size, dropout,self.no_cuda)
        n_relations = n_speakers ** 2
        self.graph_net_speaker = GraphNetwork(2 * D_e, n_relations, self.time_attn, graph_hidden_size, dropout,self.no_cuda)

        ## classification and reconstruction
        D_h = 2 * D_e + graph_hidden_size
        self.smax_fc = nn.Linear(D_h, n_classes)
        self.linear_rec = nn.Linear(D_h, adim + tdim + vdim)
        self.multihead_attn = nn.MultiheadAttention(2560, 256, dropout=dropout)
        self.linear = nn.Linear(adim + tdim + vdim, D_h)

        ## TeLU activation
        self.telu = TeLU()


    def forward(self, inputfeats, qmask, umask, seq_lengths,):
        if self.base_model == 'LSTM':
            outputs, _ = self.lstm(inputfeats[0])
            outputs = outputs.unsqueeze(2)
        elif self.base_model == 'GRU':
            outputs, _ = self.gru(inputfeats[0])
            outputs = outputs.unsqueeze(2)
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths,self.n_speakers,self.window_past, self.window_future,
         'temporal', self.no_cuda)
        assert len(edge_type_mapping) == 3
        hidden1 = self.graph_net_temporal(features, edge_index, edge_type, seq_lengths, umask)
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(outputs, qmask, seq_lengths,self.n_speakers, self.window_past, self.window_future,
         'speaker', self.no_cuda)
        assert len(edge_type_mapping) == self.n_speakers ** 2
        hidden2 = self.graph_net_speaker(features, edge_index, edge_type, seq_lengths, umask)


        hidden0 = hidden1 + hidden2

        log_prob = self.smax_fc(hidden0[0])  # [seqlen, batch, n_classes]
        rec_outputs = [self.linear_rec(hidden0[0])]

        rec_outputs[0] = rec_outputs[0].transpose(0, 1)  # Change shape to [batch, 1, adim + tdim + vdim]
        rec_outputs[0], _ = self.multihead_attn(rec_outputs[0], rec_outputs[0], rec_outputs[0])
        rec_outputs[0] = rec_outputs[0].transpose(0, 1)  # Change shape back to [1, batch, adim + tdim + vdim]
        hidden = self.telu(self.linear(rec_outputs[0]))


        return log_prob, rec_outputs, hidden


class GraphModelWithPrompt(nn.Module):
    """
    GraphModel with Prompt-based Cross-Modal Interaction

    Key Features:
    1. Cross-modal prompt fusion before sequence encoding
    2. Shared-Private feature disentanglement
    3. Router-based dynamic modality weighting
    4. Compatible with graph-based conversational dependency modeling

    Data Flow:
        audio/text/video [seq, batch, dim]
            -> CrossModalPromptFusion
            -> fused [seq, batch, adim+tdim+vdim]
            -> GRU/LSTM
            -> GNN (temporal + speaker graphs)
            -> Classification + Reconstruction
    """
    def __init__(self, base_model, adim, tdim, vdim, D_e, graph_hidden_size, n_speakers, window_past, window_future,
                 n_classes, dropout=0.5, time_attn=True, no_cuda=False, use_bn=False,
                 prompt_dim=16, n_proto=10, max_seq_len=150):
        super(GraphModelWithPrompt, self).__init__()
        self.no_cuda = no_cuda
        self.base_model = base_model
        self.adim = adim
        self.tdim = tdim
        self.vdim = vdim
        self.D_e = D_e

        # ==================== Cross-Modal Prompt Fusion ====================
        self.cross_modal_fusion = CrossModalPromptFusion(
            adim=adim, tdim=tdim, vdim=vdim, D_e=D_e,
            prompt_dim=prompt_dim, n_proto=n_proto, dropout=dropout,
            max_seq_len=max_seq_len
        )

        # ==================== Graph-based Components ====================
        # Sequence encoding: input features => 2*D_e
        self.lstm = nn.LSTM(input_size=adim + tdim + vdim, hidden_size=D_e, num_layers=2,
                           bidirectional=True, dropout=dropout)
        self.gru = nn.GRU(input_size=adim + tdim + vdim, hidden_size=D_e, num_layers=2,
                         bidirectional=True, dropout=dropout)

        self.n_speakers = n_speakers
        self.window_past = window_past
        self.window_future = window_future
        self.time_attn = time_attn

        # Graph networks for temporal and speaker relations
        n_relations = 3
        self.graph_net_temporal = GraphNetwork(2 * D_e, n_relations, self.time_attn,
                                               graph_hidden_size, dropout, self.no_cuda)
        n_relations = n_speakers ** 2
        self.graph_net_speaker = GraphNetwork(2 * D_e, n_relations, self.time_attn,
                                              graph_hidden_size, dropout, self.no_cuda)

        # Classification and reconstruction heads
        D_h = 2 * D_e + graph_hidden_size
        self.smax_fc = nn.Linear(D_h, n_classes)
        self.linear_rec = nn.Linear(D_h, adim + tdim + vdim)

        # Multi-head attention for reconstruction refinement
        total_dim = adim + tdim + vdim
        # Ensure embed_dim is divisible by num_heads
        num_heads = min(8, total_dim)  # Use at most 8 heads
        while total_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.multihead_attn = nn.MultiheadAttention(total_dim, num_heads, dropout=dropout)
        self.linear = nn.Linear(adim + tdim + vdim, D_h)

        # ==================== Modal-specific Classification Heads ====================
        self.smax_fc_a = nn.Linear(D_h, n_classes)  # Audio-specific classifier
        self.smax_fc_t = nn.Linear(D_h, n_classes)  # Text-specific classifier
        self.smax_fc_v = nn.Linear(D_h, n_classes)  # Video-specific classifier

        ## TeLU activation
        self.telu = TeLU()

    def forward(self, inputfeats, qmask, umask, seq_lengths, audio_feat=None, text_feat=None, video_feat=None,
                input_mask=None, return_disentangle=False):
        """
        Args:
            inputfeats: list containing [seq_len, batch, adim+tdim+vdim] - concatenated features
            qmask: [batch, seq_len] - speaker mask
            umask: [batch, seq_len] - utterance mask
            seq_lengths: list of conversation lengths
            audio_feat: [seq_len, batch, adim] - separate audio features (optional)
            text_feat: [seq_len, batch, tdim] - separate text features (optional)
            video_feat: [seq_len, batch, vdim] - separate video features (optional)
            input_mask: [seq_len, batch, 3] - modality availability mask (optional)
            return_disentangle: bool - whether to return disentangle features for loss calculation

        Returns:
            log_prob: [seq_len, batch, n_classes] - main classification output
            rec_outputs: list of [seq_len, batch, adim+tdim+vdim] - reconstruction output
            hidden: [seq_len, batch, D_h] - hidden representations
            (optional) shared_features, private_features: for disentangle loss
            (optional) log_prob_a, log_prob_t, log_prob_v: modal-specific predictions
        """
        input_concat = inputfeats[0]  # [seq_len, batch, adim+tdim+vdim]
        seq_len, batch, _ = input_concat.size()

        # ==================== Cross-Modal Prompt Fusion ====================
        # If separate modal features are provided, use them; otherwise split from concatenated
        if audio_feat is None:
            audio_feat = input_concat[:, :, :self.adim]
            text_feat = input_concat[:, :, self.adim:self.adim+self.tdim]
            video_feat = input_concat[:, :, self.adim+self.tdim:]

        # Apply cross-modal prompt fusion
        fused_features, shared_features, private_features = self.cross_modal_fusion(
            audio_feat, text_feat, video_feat, umask, input_mask
        )
        # fused_features: [seq_len, batch, adim+tdim+vdim]

        # ==================== Sequence Encoding (GRU/LSTM) ====================
        if self.base_model == 'LSTM':
            outputs, _ = self.lstm(fused_features)  # [seq_len, batch, 2*D_e]
            outputs = outputs.unsqueeze(2)  # [seq_len, batch, 1, 2*D_e]
        elif self.base_model == 'GRU':
            outputs, _ = self.gru(fused_features)
            outputs = outputs.unsqueeze(2)

        # ==================== Graph Neural Networks ====================
        # Temporal graph
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(
            outputs, qmask, seq_lengths, self.n_speakers,
            self.window_past, self.window_future, 'temporal', self.no_cuda
        )
        assert len(edge_type_mapping) == 3
        hidden1 = self.graph_net_temporal(features, edge_index, edge_type, seq_lengths, umask)

        # Speaker graph
        features, edge_index, edge_type, edge_type_mapping = batch_graphify(
            outputs, qmask, seq_lengths, self.n_speakers,
            self.window_past, self.window_future, 'speaker', self.no_cuda
        )
        assert len(edge_type_mapping) == self.n_speakers ** 2
        hidden2 = self.graph_net_speaker(features, edge_index, edge_type, seq_lengths, umask)

        # Combine temporal and speaker graph outputs
        hidden0 = hidden1[0] + hidden2[0]  # [seq_len, batch, D_h]

        # ==================== Classification ====================
        log_prob = self.smax_fc(hidden0)  # [seq_len, batch, n_classes]

        # Modal-specific classification (for auxiliary losses)
        log_prob_a = self.smax_fc_a(hidden0)
        log_prob_t = self.smax_fc_t(hidden0)
        log_prob_v = self.smax_fc_v(hidden0)

        # ==================== Reconstruction ====================
        rec_outputs = [self.linear_rec(hidden0)]  # [seq_len, batch, adim+tdim+vdim]

        # Apply multi-head attention for reconstruction refinement
        rec_outputs[0] = rec_outputs[0].transpose(0, 1)  # [batch, seq_len, dim]
        rec_outputs[0], _ = self.multihead_attn(rec_outputs[0], rec_outputs[0], rec_outputs[0])
        rec_outputs[0] = rec_outputs[0].transpose(0, 1)  # [seq_len, batch, dim]

        hidden = self.telu(self.linear(rec_outputs[0]))  # [seq_len, batch, D_h]

        if return_disentangle:
            return (log_prob, rec_outputs, hidden,
                    shared_features, private_features,
                    log_prob_a, log_prob_t, log_prob_v)
        else:
            return log_prob, rec_outputs, hidden
