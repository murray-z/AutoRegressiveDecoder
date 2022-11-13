import torch
import torch.nn.functional as F
from aotoregress_decode import AutoRegressDecoder


model = ""
tokenizer = ""
device = "cuda" if torch.cuda.is_available() else "cpu"


# 自定义解码器，继承AutoRegressDecoder
class MyAutoRegress(AutoRegressDecoder):
    def __init__(self):
        super(MyAutoRegress, self).__init__(
            model=model,
            maxlen=32,
            minlen=1,
            device=device,
            end_id=102,
            start_id=101
        )

    # 重写predict函数
    def predict(self, input_ids, output_ids, rtype="probas"):
        """
        :param input_ids:
        :param output_ids:
        :param rtype:
            probas: log_softmax
            logits: softmax
        :return:
        """
        input_ids = torch.cat([input_ids, output_ids], dim=1)
        output = model(*input_ids)
        last_word_hidden = output[:, -1, :]
        if rtype == "probas":
            return F.log_softmax(last_word_hidden, dim=-1)
        else:
            return F.softmax(last_word_hidden, dim=-1)

    # 自定义生成接口
    def generator(self, input_text):
        input_ids = tokenizer(input_text)

        # beam search
        res_ids = self.beam_search(input_ids, topk=3)
        res = tokenizer.decode(res_ids)

        # random sample
        res_ids = self.random_sample(input_ids, topk=3)
        res = tokenizer.decode(res_ids)

        return res


if __name__ == '__main__':
    text = "我爱北京天安门"
    gene = MyAutoRegress()
    res = gene.generator(text)
