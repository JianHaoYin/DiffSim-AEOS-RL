import torch
import torch.nn as nn

class PositionEmbedding(nn.Module):
    def __init__(self, dim_model,
                max_len=10000,
                device='cuda' if torch.cuda.is_available() else 'cpu'):
        super(PositionEmbedding, self).__init__()


        position = torch.arange(0, max_len).float()

        _2i = torch.arange(0, dim_model, step=2, device=device).float()

        self.encoding = torch.zeros((max_len, dim_model), device=device).float()
        #torch.sin 处理的是一个tensor里所有的元素
        #这里用一个维度上的position和另一个维度上的2i做运算，得到一个二维矩阵，获得了每个position对应的偶数行模版
        #在输出端把偶数的dim维度切出来，position维度不变，把算完的sin和cos放进去
        self.encoding[:, 0::2] = torch.sin(position / (10000 ** (2 * _2i / dim_model)))
        self.encoding[:, 1::2] = torch.cos(position / (10000 ** (2 * _2i / dim_model)))



    def forward(self, x):
        '''
        Args:
            x: Tensor, shape [batch_size, seq_len]
        Returns:
            Tensor, shape [batch_size, seq_len, embedding_dim]
        '''

        batch_size, seq_len = x.size()

        #位置编码只和位置序号相关，提前算好，根据seq_len截取
        return self.encoding[:seq_len, :].unsqueeze(0).repeat(batch_size, 1, 1)
    
if __name__ == "__main__":
    input = torch.zeros(4, 10)
    pos_emb = PositionEmbedding(512)
    output = pos_emb(input)
    print(output.shape)
