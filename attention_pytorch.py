from torch import nn, from_numpy
import numpy as np
import math


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()  # 初始化 nn.Module 的所有功能,调用父类的构造方法
        self.d_model = d_model  # 把传进来的 d_model 存成实例变量，供模块内部使用
        self.h = h  # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h  # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv
        self.w_o = nn.Linear(
            d_model, d_model,
            bias=False)  # Wo 多头注意力机制（Multi-Head Attention） 的最后一步，用来把多头的输出合并起来
        self.dropout = nn.Dropout(dropout)
        """
        每个头都分别计算一套注意力，然后得到形状:
        (batch_size, seq_len, d_k)   ← 每个头的输出
        有 h 个头, 所以最后会把所有头拼在一起变成:
        (batch_size, seq_len, h * d_k)   ← 也就是 d_model
        但是拼起来还不够,我们还需要用一个线性变换把这个合成的结果,再投影回模型空间，让它适配后面的网络层。
        (batch_size, seq_len, d_model)拼起来 → self.w_o = nn.Linear(d_model, d_model, bias=False)
        这个 w_o 就是那个投影矩阵，作用类似于“融合各个头的信息”。
        w_o 的作用：把多个注意力头的输出拼接后，通过线性变换整合成模型继续处理的统一特征空间
        拼接：得到形状 (batch, seq_len, d_model)
        w_o：再来一个 Linear(d_model, d_model)，不是改变维度，而是改变信息的表达方式
        """

    @staticmethod  #放在类里面的普通函数，但它不依赖类的状态（属性或实例），只是一段逻辑代码的集合
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # Just apply the formula from the paper
        # query.shape == (batch_size, num_heads, seq_len, d_k)
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(
            d_k)  #缩放点积注意力（scaled dot-product attention）
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(
            dim=-1)  # (batch, h, seq_len, seq_len) # Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        #用 nn.Linear 得到真正的 Q/K/V 向量（需要参数 Wq/Wk/Wv）
        query = self.w_q(
            q)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        key = self.w_k(
            k)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        value = self.w_v(
            v)  # (batch, seq_len, d_model) --> (batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h,
                           self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h,
                       self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h,
                           self.d_k).transpose(1, 2)
        """
        都是 PyTorch 中改变张量形状的方法
        1. view()

        必须要求张量是**连续的（contiguous）**内存块。

        如果张量不是连续的，调用 view() 会报错。

        性能更快，因为它只是改变视图，不复制数据。

        如果张量非连续，可以先调用 .contiguous() 让它变连续，再用 view()。

        2. reshape()

        更灵活，可以处理非连续的张量。

        如果原始张量不是连续的，reshape() 会返回一个新张量的副本（拷贝数据）。

        用起来更方便，避免出错。

        """
        # Calculate attention 输入已经是 Q/K/V，用于计算注意力矩阵和加权输出
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout)

        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1,
                                                self.h * self.d_k)
        """
        转置 x.transpose(1, 2)：  交换第1和第2维，也就是把 (batch, h, seq_len, d_k) 转成 (batch, seq_len, h, d_k) 这样，序列长度维度到第2位，头数到第3位。
        调用 contiguous()：因为转置只是改变了视图，内存不连续，contiguous() 让数据内存连续，方便下一步 view。

        调用 view(x.shape[0], -1, self.h * self.d_k)：这里把 (batch, seq_len, h, d_k) 变成 (batch, seq_len, d_model)
        -1 自动推断为 seq_len，self.h * self.d_k 就是 d_model
        当用 .view() 或 .reshape() 调整张量形状时，给某个维度写 -1，PyTorch 会根据其他维度和总元素数，自动计算这个维度应该是多少。
        """
        # Multiply by Wo
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)
