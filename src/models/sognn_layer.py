import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch.nn import Linear, LayerNorm
from torch_sparse import SparseTensor, spspmm

class SOGNNConv(MessagePassing):
    def __init__(
            self, 
            in_channels: int, 
            out_channels: int, 
            distant_param: int=5, 
            chunk_size: int=4
    ):
        super(SOGNNConv, self).__init__('mean')  # 'add', 'sum', 'mean', 'min', 'max'

        self.out_channels = out_channels
        self.lin = Linear(in_channels, out_channels)
        self.norm = LayerNorm(out_channels)
        # 用chunk_size来减少参数，一个门控制chunk_size个神经元
        self.chunk_size = chunk_size

        # 如果chunk_size不能整除out_channels，则让chunk_size = out_channels
        if not out_channels % chunk_size == 0:
            self.chunk_size = out_channels

        self.gat = Linear(3*out_channels, self.chunk_size)


    def forward(self, x, edge_index: Tensor, edge_index_distant: SparseTensor):
        # Aggregate
        x = self.lin(x)
        edge_index_close = edge_index
        # edge_index_distant = self.get_distant_adjacent_matrix(edge_index)
        m_n = self.propagate(edge_index_close, x=x)
        m_d = self.propagate(edge_index_distant, x=x)
        # Combine
        gat_raw = F.softmax(self.gat(torch.cat((x, m_n, m_d), dim=1)), dim=-1)
        gat = torch.cumsum(gat_raw, dim=-1).repeat_interleave(repeats=int(self.out_channels/self.chunk_size), dim=1)
        out = (x + m_n) * (1 - gat) + (x + m_n + m_d.flip(dims=[1])) * gat
        out = self.norm(out)

        return out


    @classmethod
    def get_distant_adjacent_matrix(
            self, 
            edge_index: Tensor, 
            mode="mean", 
            temperatur=1000,
            r=5
        ) -> SparseTensor:
        '''
        根据邻接矩阵的幂 -> 节点间距离为幂的路径条数, 筛选易被squash的点 <- 路径少
        '''
        mode_list = ['mean']
        assert mode in mode_list, "unvalid mode selected, pleace choose one mode in {}.".format(mode_list)

        assert r > 1, "Distance should be greater than 1"

        with torch.no_grad():
            # 计算邻接矩阵的r次方，以稀疏矩阵的方式运算
            # 利用最小空间存储sparse tensor
            nodes = torch.max(edge_index).item() + 1
            value_edge_index = torch.ones((edge_index.shape[1])).to(edge_index.device)

            edge_index_distant = edge_index.clone()
            value_edge_index_distant = value_edge_index.clone()
            for _ in range(r - 1):
                edge_index_distant, value_edge_index_distant = spspmm(
                    edge_index_distant,
                    value_edge_index_distant,
                    edge_index,
                    value_edge_index,
                    nodes, nodes, nodes, coalesced=True
                )

            # 用不同的规则构造远距离squash点的邻接矩阵
            # 1. 计算切分点
            ei = edge_index[0]
            ei_ = torch.cat([ei[0:1], ei[:-1]])

            cutpoints = torch.nonzero(ei - ei_).squeeze().tolist()
            cutpoints = [0] + cutpoints + [ei.shape[0]]

            # 2. 切分spt
            adj_raw = [(edge_index_distant[:, start:end], value_edge_index_distant[start:end]) 
                        for start, end in zip(cutpoints[:-1], cutpoints[1:])]
            
            # 3. 对spt进行筛选
            if mode=="mean":
            # 筛选||大于零|小于平均值||的邻居的index ———— Sparse Tensor
                mean_value = [torch.mean(value) / temperatur for data, value in adj_raw]
                adj_selected_raw = [data[:, value<mean] for (data, value), mean in zip(adj_raw, mean_value)]

            # 4. 将筛选出来的spt拼接成最终的spt
            adj_selected = torch.cat(adj_selected_raw, dim=-1)
            spt_distant = SparseTensor.from_edge_index(adj_selected, sparse_sizes=(nodes, nodes))


        return spt_distant


