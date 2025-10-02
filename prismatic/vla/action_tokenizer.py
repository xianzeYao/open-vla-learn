"""
action_tokenizer.py

Extension class; wraps base LLM/VLM tokenizer with logic to discretize and tokenize continuous robot actions.
扩展模块：在原有 LLM/VLM tokenizer 之上加入动作离散化逻辑，将连续控制信号映射为离散 token。
"""

from typing import List, Union

import numpy as np
from transformers import PreTrainedTokenizerBase

# 动作 tokenizer：包装基础 tokenizer 并提供动作离散化/反解码能力


class ActionTokenizer:
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, bins: int = 256, min_action: int = -1, max_action: int = 1
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
                          需要扩展的基础 tokenizer（通常来自语言模型）。
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
                      每个动作维度划分的离散区间数量，默认均匀划分。
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
                           动作的最小取值，用于裁剪和确定区间下界。
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
                           动作的最大取值，用于裁剪和确定区间上界。
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        # 统一在 [-1,1] 区间划分等宽区间，每个区间对应一个离散动作 token
        self.bins = np.linspace(min_action, max_action, self.n_bins) # 得到self.n_bins个点，self.n_bins-1个区间
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0 

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        # 约定：把词表末尾的若干 token 让给动作编码，因此记录起始下标方便外部构造 mask
        self.action_token_begin_idx: int = int(
            self.tokenizer.vocab_size - self.n_bins - 1) # 留最后几个位置作为token

    def __call__(self, action: np.ndarray) -> Union[str, List[str]]:
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:]).
        将连续动作剪切并映射到词表末端预留的动作 token 上。
        """
        # 将连续动作裁剪到合法范围，使用 numpy.digitize 拿到离散 bin 序号
        action = np.clip(action, a_min=float(self.min_action),
                         a_max=float(self.max_action)) # action会是浮点数矩阵
        discretized_action = np.digitize(action, self.bins) #将连续的action值映射到离散区间bins中，得到的是对应区间的编号

        # Handle single element vs. batch
        # 单步动作直接 decode；批量动作使用 batch_decode，得到同长度的 token 序列
        if len(discretized_action.shape) == 1: #这里的decoder代表的是llm将tokenid->可读字符串
            # self.tokenizer.vocab_size - discretized_action 得到的就是对应在llm中词表token的id
            # 数字倒叙是因为大模型的词表末尾的token是最少用的
            return self.tokenizer.decode(list(self.tokenizer.vocab_size - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer.vocab_size - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray: #将token对应的token id转为动作向量
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        将离散动作 token id 还原成连续动作值。
        """
        # 将动作 token 反向映射到 bin 序号：词表末尾 token => 低序号即词表最后一个词->区间序号1
        discretized_actions = self.tokenizer.vocab_size - action_token_ids
        # digitize 返回区间索引为 1..n，这里减一并 clip，避免越界到最后一个中心点之外
        # 即256个动作，256个端点值，255个区间，255个区间中点，action的区间序号1-256，-1后为0-255，右有索引溢出可能故再clip
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions] # 序列号对应一个区间，区间中点即代表这个区间的动作

    @property
    def vocab_size(self) -> int:
        # 暴露可用的动作 token 数量，供上层推理构造输出空间
        return self.n_bins
