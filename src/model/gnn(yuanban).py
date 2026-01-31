import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, TransformerConv, GATConv, MessagePassing
from torch.nn import BatchNorm1d, LayerNorm
from torch_geometric.nn import GATv2Conv



class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x, edge_attr

#ajejc 9.10原版
class GraphTransformer(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super(GraphTransformer, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(TransformerConv(in_channels=in_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=hidden_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(TransformerConv(in_channels=hidden_channels, out_channels=out_channels//num_heads, heads=num_heads, edge_dim=in_channels, dropout=dropout,))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, adj_t, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=adj_t, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=adj_t, edge_attr=edge_attr)
        return x, edge_attr



class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super(GAT, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x,edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr



# ajejc 9.9 H1:69.90
# class ReaRevConv(MessagePassing):
#     def __init__(self, in_channels, out_channels, edge_dim, dropout=0.5):
#         super(ReaRevConv, self).__init__(aggr='mean')  # 使用mean聚合
#         self.node_lin = torch.nn.Linear(in_channels, out_channels)  # 线性层处理节点特征
#         self.edge_lin = torch.nn.Linear(edge_dim, out_channels)  # 线性层处理边特征
#         self.dropout = dropout

#     def forward(self, x, edge_index, edge_attr):
#         # 分别处理节点特征和边特征
#         x = self.node_lin(x)  # 处理节点特征
#         edge_weight = self.edge_lin(edge_attr)  # 处理边特征作为权重
        
#         # 使用边特征作为权重进行消息传递
#         return self.propagate(edge_index, x=x, edge_weight=edge_weight)

#     def message(self, x_j, edge_weight):
#         # 使用边特征加权邻居节点特征
#         return x_j * edge_weight

#     def update(self, aggr_out):
#         return F.relu(aggr_out)  # 激活输出


# class ReaRev(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads):
#         super(ReaRev, self).__init__()
#         self.convs = torch.nn.ModuleList()

#         # 第一层
#         self.convs.append(ReaRevConv(in_channels, hidden_channels, edge_dim=hidden_channels, dropout=dropout))
#         self.bns = torch.nn.ModuleList()
#         self.bns.append(BatchNorm1d(hidden_channels))

#         # 中间层
#         for _ in range(num_layers - 2):
#             self.convs.append(ReaRevConv(hidden_channels, hidden_channels, edge_dim=hidden_channels, dropout=dropout))
#             self.bns.append(BatchNorm1d(hidden_channels))

#         # 最后一层
#         self.convs.append(ReaRevConv(hidden_channels, out_channels, edge_dim=hidden_channels, dropout=dropout))
#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, x, edge_index, edge_attr):
#         for i, conv in enumerate(self.convs[:-1]):
#             x = conv(x, edge_index, edge_attr)
#             x = self.bns[i](x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index, edge_attr)
#         return x, edge_attr



class ReaRevConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim, num_heads=4, dropout=0.5):
        super(ReaRevConv, self).__init__(aggr='mean')
        self.node_lin = torch.nn.Linear(in_channels, out_channels)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(edge_dim, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, out_channels)
        )
        self.num_heads = num_heads
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr):
        x = self.node_lin(x)
        edge_weight = self.edge_mlp(edge_attr)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        # 调整attention_score的维度，确保与x_j相乘时维度一致
        attention_score = F.softmax(edge_weight.view(edge_weight.size(0), self.num_heads, -1), dim=-1)
        # 通过多头注意力对x_j进行加权
        return (x_j.view(x_j.size(0), self.num_heads, -1) * attention_score).view(x_j.size(0), -1)




class ReaRev(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads):
        super(ReaRev, self).__init__()
        self.convs = torch.nn.ModuleList()

        # 第一层
        self.convs.append(ReaRevConv(in_channels, hidden_channels, edge_dim=hidden_channels, dropout=dropout))
        self.bns = torch.nn.ModuleList()
        self.bns.append(BatchNorm1d(hidden_channels))

        # 中间层
        for _ in range(num_layers - 2):
            self.convs.append(ReaRevConv(hidden_channels, hidden_channels, edge_dim=hidden_channels, dropout=dropout))
            self.bns.append(BatchNorm1d(hidden_channels))

        # 最后一层
        self.convs.append(ReaRevConv(hidden_channels, out_channels, edge_dim=hidden_channels, dropout=dropout))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-2]):  # 只在前几层不使用跳层连接
            x = conv(x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # 在最后两层进行跳层连接
        x_mid = x  # 保存中间层特征
        x = self.convs[-2](x, edge_index, edge_attr)
        x = self.bns[-2](x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_attr)
        x = x + x_mid  # 跳层连接在最后两层进行
        return x, edge_attr
    


#平均72.113025最高
# class GATv2Network(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads=4, dropout=0.6, edge_dim=None):
#         super(GATv2Network, self).__init__()
#         self.convs = torch.nn.ModuleList()

#         # 确保 edge_dim 是边特征的维度，不是 None
#         self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=False, edge_dim=in_channels))
#         self.bns = torch.nn.ModuleList()
#         self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
#         for _ in range(num_layers - 2):
#             self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=num_heads, concat=False, edge_dim=in_channels))
#             self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
#         self.convs.append(GATv2Conv(hidden_channels, out_channels, heads=num_heads, concat=False, edge_dim=in_channels))
#         self.dropout = dropout

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, x, edge_index, edge_attr):
#         for i, conv in enumerate(self.convs[:-1]):
#             residual = x  # 保存输入以便残差连接
#             x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
#             x = self.bns[i](x)
#             x = F.relu(x + residual)  # 使用残差连接
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
#         return x, edge_attr


#ajejc 最高72.727
# class GATv2Network(torch.nn.Module):
#     def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads=4, dropout=0.6, edge_dim=None):
#         super(GATv2Network, self).__init__()
#         self.convs = torch.nn.ModuleList()

#         # 第一个 GATv2Conv 层
#         self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=False, edge_dim=in_channels))
#         self.bns = torch.nn.ModuleList()
#         self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

#         # 中间层的 GATv2Conv
#         for _ in range(num_layers - 2):
#             self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=num_heads, concat=False, edge_dim=in_channels))
#             self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
#         # 最后一个 GATv2Conv 层
#         self.convs.append(GATv2Conv(hidden_channels, out_channels, heads=num_heads, concat=False, edge_dim=in_channels))
#         self.dropout = dropout

#         # 当输入和输出维度不匹配时，使用线性变换匹配残差连接
#         if in_channels != hidden_channels:
#             self.residual_lin = torch.nn.Linear(in_channels, hidden_channels)
#         else:
#             self.residual_lin = None

#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#         for bn in self.bns:
#             bn.reset_parameters()

#     def forward(self, x, edge_index, edge_attr):
#         for i, conv in enumerate(self.convs[:-1]):
#             # 保存输入作为残差
#             if self.residual_lin is not None and i == 0:
#                 residual = self.residual_lin(x)  # 匹配维度的线性变换
#             else:
#                 residual = x

#             x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
#             x = self.bns[i](x)
#             x = F.relu(x + residual)  # 残差连接
#             x = F.dropout(x, p=self.dropout, training=self.training)

#         # 最后一层不使用残差连接
#         x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
#         return x, edge_attr

#ajejc 最高73.09582
class GATv2Network(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, num_heads=4, dropout=0.6, edge_dim=None):
        super(GATv2Network, self).__init__()
        self.convs = torch.nn.ModuleList()

        # 第一个 GATv2Conv 层
        self.convs.append(GATv2Conv(in_channels, hidden_channels, heads=num_heads, concat=False, edge_dim=in_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))

        # 中间层的 GATv2Conv
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels, heads=num_heads, concat=False, edge_dim=in_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        
        # 最后一个 GATv2Conv 层
        self.convs.append(GATv2Conv(hidden_channels, out_channels, heads=num_heads, concat=False, edge_dim=in_channels))
        self.dropout = dropout

        # 当输入和输出维度不匹配时，使用线性变换匹配残差连接
        if in_channels != hidden_channels:
            self.residual_lin = torch.nn.Linear(in_channels, hidden_channels)
        else:
            self.residual_lin = None

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            # 保存输入作为残差
            if self.residual_lin is not None and i == 0:
                residual = self.residual_lin(x)  # 匹配维度的线性变换
            else:
                residual = x

            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x + residual)  # 残差连接
            x = F.dropout(x, p=self.dropout, training=self.training)

        # 最后一层不使用残差连接
        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr

    def message(self, x_j, edge_attr):
        combined_features = torch.cat([x_j, edge_attr], dim=-1)
        mlp = torch.nn.Sequential(
            torch.nn.Linear(combined_features.size(-1), x_j.size(-1)),
            torch.nn.ReLU(),
            torch.nn.Linear(x_j.size(-1), x_j.size(-1))
        )
        attention_weights = F.softmax(edge_attr, dim=-1)
        return mlp(combined_features * attention_weights)







    

load_gnn_model = {
    'gcn': GCN,
    'gat': GAT,
    'gt': GraphTransformer,
    'rr': ReaRev,
    'gat2': GATv2Network
}
