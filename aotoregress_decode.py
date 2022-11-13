# coding:utf-8

import torch
import numpy as np


class AutoRegressDecoder(object):
    """
    自回归解码器，包含beam search和random sample
    """
    def __init__(self, start_id, end_id, model, maxlen, minlen, device):
        """
        :param start_id: 解码器输入起始标记id
        :param end_id: 解码器终止id
        :param model: seq2seq模型
        :param maxlen: 输出最长值，如果达到该值没有遇到 end_id 则提前结束预测
        :param minlen: 输出最小长度值
        :param device: 模型、输入所在设备
        """
        self.stari_id = start_id
        self.end_id = end_id
        self.model = model
        self.maxlen = maxlen
        self.minlen = minlen
        self.device = device
        self.model.to(device)
        self.first_output_ids = torch.tensor([[self.stari_id]], dtype=torch.long).to(device)

    def predict(self, input_ids, output_ids, rtype="probas"):
        """用户根据自己模型自定义预测函数
           返回：
            probas: softmax之后的返回值
            logits: log_softmax之后返回值
        """
        raise NotImplementedError

    def beam_search(self, input_ids, topk, min_ends=1):
        """
        beam search解码方法，返回搜索到的一条最优解
        :param input_ids: 模型输入编码后的id, tensor
        :param topk: beam search size
        :return:
        """
        # 将输入和模型放到一个设备
        input_ids = [i.to(self.device) for i in input_ids]
        output_ids, output_scores = self.first_output_ids, torch.tensor(1, dtype=torch.float).to(self.device)
        for step in range(self.maxlen):
            scores = self.predict(input_ids, output_ids, rtype="logits")

            # 获得词典大小
            vocab_size = scores.shape[-1]

            # 第一次预测之后，将输入重复topk次
            if step == 0:
                input_ids = [i[0].repeat(topk, 1) for i in input_ids]

            # 累积得分
            scores = output_scores.reshape((-1, 1)) + scores
            scores = scores.view(-1)

            # 取topk个最大值
            values, indices = torch.topk(scores, topk)

            # 找出最大值topk所在位置信息
            indices_1 = (indices // vocab_size)  # 第几条序列
            indices_2 = (indices % vocab_size).reshape((-1, 1))  # 序列最大得分字在词典中索引

            # 将最大值合并到输出序列
            output_ids = torch.cat([output_ids[indices_1], indices_2], dim=1)

            # 更新得分
            output_scores = scores[indices]

            # 统计是否出现终止符号
            end_counts = torch.sum(output_ids == self.end_id, dim=1)

            # 判断是否达到了最短长度
            if output_ids.shape[-1] >= self.minlen:
                best_one = torch.argmax(output_scores)
                # 最优路径已经到达终止符号
                if end_counts[best_one] == min_ends:
                    return output_ids[best_one]
                else:
                    # 未达到终止符号
                    flag = (end_counts < min_ends)
                    # 存在未完成序列
                    if not flag.all():
                        input_ids = [i[flag] for i in input_ids]
                        output_ids = output_ids[flag]
                        output_scores = output_scores[flag]
                        topk = flag.sum()
        # 达到长度直接输出
        return output_ids[torch.argmax(output_scores)]

    def random_sample(self, input_ids, n, topk, topp, min_ends=1):
        results = []

        input_ids = [i.to(self.device) for i in input_ids]
        output_ids = self.first_output_ids

        for step in range(self.maxlen):
            probas = self.predict(input_ids, output_ids, rtype="probas")
            # 确保归一化
            probas /= torch.sum(probas, dim=1, keepdim=True)
            # 第一步之后对输入值进行复制n份
            if step == 0:
                input_ids = [i[0].repeat(n, 1) for i in input_ids]
                probas = probas.repeat(n, 1)

            if topk is not None:
                # 取topk的索引
                k_values, k_indices = torch.topk(probas, topk, dim=-1)
                if torch.__version__.split("+")[0] >= '1.9.0':
                    probas = torch.take_along_dim(probas, k_indices, dim=1)
                else:
                    probas = torch.tensor(np.take_along_axis(probas.detach().cpu().numpy(),
                                                             k_indices.cpu().numpy(), axis=1)).to(self.device)
                # 归一化
                probas /= torch.sum(probas, dim=1, keepdim=True)

            if topp is not None:
                # 降序排列，取索引
                p_indices = torch.argsort(probas, dim=1, descending=True)
                if torch.__version__.split("+")[0] >= '1.9.0':
                    probas = torch.take_along_dim(probas, p_indices, dim=1)
                else:
                    probas = torch.tensor(np.take_along_axis(probas.detach().cpu().numpy(),
                                                             p_indices.cpu().numpy(), axis=1)).to(self.device)
                # 累积概率
                cumsum_probas = torch.cumsum(probas, dim=1)
                # 标记超过topp的位置，由于超过topp的第一个位置需要保留
                # 采用roll将尾部数据移到第一个位置
                flag = torch.roll(cumsum_probas >= topp, 1, dims=1)
                flag[:, 0] = False
                # 将尾部概率较小的值置零
                probas[flag] = 0
                # 概率归一化
                probas /= torch.sum(probas, dim=1, keepdim=True)

            # 采样函数，按照概率进行采样
            sample_fun = lambda p: np.random.choice(len(p), p=p)
            sample_ids = np.apply_along_axis(sample_fun, 1, probas.detach().cpu().numpy())
            sample_ids = torch.tensor(sample_ids.reshape((-1, 1))).to(self.device)

            if topp is not None:
                if torch.__version__.split("+")[0] >= '1.9.0':
                    sample_ids = torch.take_along_dim(p_indices, sample_ids, dim=1)
                else:
                    sample_ids = np.take_along_axis(p_indices.detach().cpu().numpy(),
                                                    sample_ids.detach().cpu().numpy(),
                                                    axis=1)
                    sample_ids = torch.tensor(sample_ids).to(self.device)

            if topk is not None:
                if torch.__version__.split("+")[0] >= '1.9.0':
                    sample_ids = torch.take_along_dim(k_indices, sample_ids, dim=1)
                else:
                    sample_ids = np.take_along_axis(k_indices.detach().cpu().numpy(),
                                                    sample_ids.detach().cpu().numpy(),
                                                    axis=1)
                    sample_ids = torch.tensor(sample_ids).to(self.device)

            output_ids = torch.cat([output_ids, sample_ids], dim=1)

            # 统计出现结束符号的数目
            end_counts = torch.sum(output_ids==self.end_id, dim=1)

            # 输出长度大于最短长度
            if output_ids.shape[1] >= self.minlen:
                # 标记已完成序列
                flag = (end_counts == min_ends)
                if flag.any():
                    for ids in output_ids[flag]:
                        results.append(ids)
                    # 标记未完成序列
                    flag = (flag==False)
                    # 仅保留未完成序列
                    input_ids = [i[flag] for i in input_ids]
                    output_ids = output_ids[flag]
                    end_counts = end_counts[flag]
                    if len(output_ids) == 0:
                        break

        # 达到最大长度仍然有未完成序列
        for ids in output_ids:
            results.append(ids)

        return results


if __name__ == '__main__':
    print(torch.__version__)
    print(torch.__version__ > "1.9.0")





