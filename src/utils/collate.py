from torch_geometric.data import Batch
import torch

def collate_fn(original_batch):
    # 获取目标设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 将每个图形数据移动到目标设备
    for item in original_batch:
        item['graph'].x = item['graph'].x.cpu()  # 确保在 CPU
        item['graph'].edge_attr = item['graph'].edge_attr.cpu()  # 确保在 CPU
        item['graph'].edge_index = item['graph'].edge_index.cpu()  # 确保在 CPU

    # 创建批量
    batch = {}
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    if 'graph' in batch:
        batch['graph'] = Batch.from_data_list(batch['graph'])
    return batch
