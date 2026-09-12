"""观测编码器：视觉 (DINOv2)、语言 (SigLIP)、本体感知 (MLP)。

三者统一输出 d_model 维的 token 序列，拼成一个 context 张量喂给 FM denoiser 做
cross-attention。视觉和语言的骨干网络都冻结 —— 演示数据只有几千条轨迹，
微调 ViT 必然过拟合，而且 8GB 显存也放不下反传所需的激活。
"""
from __future__ import annotations

import torch
import torch.nn as nn

DINOV2_MODEL = "vit_small_patch14_dinov2.lvd142m"
SIGLIP_MODEL = "google/siglip-base-patch16-224"


class VisualEncoder(nn.Module):
    """冻结的 DINOv2-Small，输出每个 patch 的 token。

    输入  imgs: (B, T, 3, H, W)   T 帧堆叠的观测窗口
    输出  tokens: (B, N_patches, d_model)

    为什么用 patch token 而不是 CLS token：操作任务需要知道物体**在哪**，
    空间信息全在 patch token 里，CLS 池化会把它丢掉。

    为什么时间维取平均而不是拼接：拼接会让 context 长度翻 T 倍，cross-attention
    的开销也翻倍；短观测窗口(T=2)主要是为了让策略感知速度，平均已经够用。
    """

    def __init__(self, d_model: int = 256, img_size: int = 126, freeze: bool = True):
        super().__init__()
        import timm

        # DINOv2 的 patch 是 14x14，img_size 必须是 14 的倍数
        assert img_size % 14 == 0, f"DINOv2 patch=14, img_size 必须被 14 整除, got {img_size}"
        self.img_size = img_size
        self.dino = timm.create_model(
            DINOV2_MODEL, pretrained=True, num_classes=0, img_size=img_size
        )
        self.frozen = freeze
        if freeze:
            self.dino.eval()
            for p in self.dino.parameters():
                p.requires_grad = False

        self.embed_dim = self.dino.embed_dim  # 384
        self.n_patches = (img_size // 14) ** 2
        self.proj = nn.Linear(self.embed_dim, d_model)

    def train(self, mode: bool = True):
        """冻结时始终保持骨干网络在 eval 模式（关掉 dropout 等）。"""
        super().train(mode)
        if self.frozen:
            self.dino.eval()
        return self

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # 训练时可以喂预先算好的 patch token 而不是像素（见
        # scripts/build_feature_cache.py）。骨干冻结、管线里没有图像增强，
        # 所以这两条路径**数值等价**，只是省掉了约 88% 的训练时间。
        # 靠维数分派：像素是 (B,T,3,H,W) 五维，缓存特征是 (B,T,N,384) 四维。
        if imgs.ndim == 4:
            assert imgs.shape[-1] == self.embed_dim, (
                f"缓存特征的最后一维应为 {self.embed_dim}, got {imgs.shape[-1]}")
            return self.proj(imgs.mean(dim=1))

        B, T = imgs.shape[:2]
        flat = imgs.reshape(B * T, *imgs.shape[2:])

        ctx = torch.no_grad() if self.frozen else torch.enable_grad()
        with ctx:
            feats = self.dino.forward_features(flat)      # (B*T, 1+N, 384)
        feats = feats[:, self.dino.num_prefix_tokens:, :]  # 去掉 CLS / register token

        feats = feats.reshape(B, T, self.n_patches, self.embed_dim).mean(dim=1)
        return self.proj(feats)                            # (B, N_patches, d_model)


class LanguageEncoder(nn.Module):
    """冻结的 SigLIP 文本塔，输出单个语言 token，并缓存已编码过的指令。

    输入  instructions: list[str]，长度 B
    输出  tokens: (B, 1, d_model)

    任务指令在整个数据集里只有少数几种固定字符串，每步重新编码纯属浪费，
    所以按字符串缓存。空串 "" 作为 CFG 的 null condition，也走同一条缓存路径。
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        from transformers import AutoTokenizer, SiglipTextModel, logging as hf_logging

        # 只取文本塔，checkpoint 里的视觉塔权重会被报成 UNEXPECTED —— 属正常
        prev_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()

        self.tokenizer = AutoTokenizer.from_pretrained(SIGLIP_MODEL)
        self.text_model = SiglipTextModel.from_pretrained(SIGLIP_MODEL)
        hf_logging.set_verbosity(prev_verbosity)
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.text_model.eval()

        self.embed_dim = self.text_model.config.hidden_size  # 768
        self.proj = nn.Linear(self.embed_dim, d_model)
        # 缓存放在 buffer 之外的普通 dict，不进 state_dict
        self._cache: dict[str, torch.Tensor] = {}

    def train(self, mode: bool = True):
        super().train(mode)
        self.text_model.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.proj.weight.device

    @torch.no_grad()
    def _encode(self, texts: list[str]) -> torch.Tensor:
        # SigLIP 的文本塔要求定长 padding 到 64（训练时就是这么做的）
        batch = self.tokenizer(
            texts, padding="max_length", max_length=64,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        return self.text_model(**batch).pooler_output  # (n, 768)

    def forward(self, instructions: list[str]) -> torch.Tensor:
        missing = [s for s in set(instructions) if s not in self._cache]
        if missing:
            embs = self._encode(missing)
            for s, e in zip(missing, embs):
                self._cache[s] = e

        stacked = torch.stack([self._cache[s] for s in instructions]).to(self.device)
        return self.proj(stacked).unsqueeze(1)  # (B, 1, d_model)

    def clear_cache(self) -> None:
        self._cache.clear()


class ProprioEncoder(nn.Module):
    """本体感知 MLP：关节角 + 夹爪状态 -> 单个 token。

    输入  proprio: (B, T, proprio_dim)
    输出  tokens: (B, 1, d_model)

    只用最后一帧：机器人当前构型是马尔可夫的，历史信息主要用于估计速度，
    而速度已经隐含在视觉的多帧输入里了。
    """

    def __init__(self, proprio_dim: int = 9, d_model: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(proprio_dim, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU(),
            nn.Linear(128, d_model),
        )

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        return self.net(proprio[:, -1, :]).unsqueeze(1)  # (B, 1, d_model)
