"""ManiSkill3 演示数据集：把重放后的 .h5 轨迹切成 (观测窗口, 动作块) 样本对。

数据来源说明：官方下载的 demo 里 obs/ 是空的，需要先用 scripts/replay_demos.sh
重放渲染，得到 trajectory.rgb.pd_joint_pos.physx_cpu.h5。

关键设计 —— 索引与数据分离：
构造时只扫描每条轨迹的长度、建立 (episode_id, start_t) 的扁平索引，
图像不预先读进内存（1000 条 × 75 帧 × 128²×3 ≈ 3.7GB/任务，三个任务放不下）。
__getitem__ 时才按需从 h5 读取对应的窗口。
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet 统计量 —— DINOv2 预训练时用的就是这个，必须一致
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# 演示重放时转换到的控制模式。改这里要同步重跑 scripts/replay_demos.sh。
# 用末端增量而非演示原生的 pd_joint_pos：后者 98.4% 的动作方差只是"手臂当前在哪"，
# 任务信号只占 1.6%，信噪比太低（见 docs/debugging.md）。
CONTROL_MODE = "pd_ee_delta_pose"

# 指令是多任务下唯一区分任务的信号，所以措辞要让动词承重：Pick / Push / Pull
# 三个任务的场景几乎一样（桌面 + 一个方块 + 一个目标点），差别全在这个动词上。
# 这也是"语言到底有没有被用上"那个消融能成立的前提，见 scripts/language_ablation.py。
TASK_INSTRUCTIONS = {
    "PickCube-v1": "Pick up the cube and move it to the goal position",
    "PushCube-v1": "Push the cube to the goal region",
    "PullCube-v1": "Pull the cube to the goal region",
    "StackCube-v1": "Stack the red cube on the green cube",
    "PegInsertionSide-v1": "Insert the peg into the hole",
}


# 每个任务的"目标位置"取自场景里的哪个 actor。
#
# 这是多任务的关键设计。README 曾把它写成一条无解的架构冲突：PickCube 的目标是
# 相机看不见的随机标记（必须显式给），StackCube / PegInsertion 的目标是场景实物
# （演示数据里根本没有 goal_pos 字段），于是"统一观测空间"与"提供显式目标"
# 不可兼得。但冲突只存在于**目标从哪来**，不存在于**目标是什么** ——
# 三个任务的目标都是一个 3 维世界坐标：物体最终该到哪。
#
# 先前试过的替代方案是"目标槽 + 有效位"（13 维，没有目标的任务填零并置 valid=0）。
# 它把观测空间统一了，但**没用**：三个任务里有两个的目标槽恒为零，proprio 编码器
# 学会了不看它，多任务 PickCube 退化到"抓取率 90%、成功率 6%"——
# 与单任务完全不给目标时的 6% 一模一样（见 docs/debugging.md 第五轮）。
#
# 值全部来自 `env_states/actors/<name>` 的前 3 维（世界坐标位置），
# 与评测时从环境实时读到的逐位一致，已对拍验证。
TASK_GOAL_ACTOR = {
    "PickCube-v1": "goal_site",      # 每局随机的半透明标记，相机里不可见
    "PushCube-v1": "goal_region",    # 画在桌面上的目标区域，相机里可见
    "PullCube-v1": "goal_region",
    "StackCube-v1": "cubeB",         # 目标是"叠到 cubeB 上"，所以目标就是 cubeB 的位置
}


class ManiskillDataset(Dataset):
    """每个样本：
        rgb         (obs_horizon, 3, img_size, img_size)  float32, 已归一化
        proprio     (obs_horizon, proprio_dim)            float32
        action      (act_horizon, act_dim)                float32
        instruction str
    """

    def __init__(
        self,
        h5_path: str | Path,
        task: str,
        obs_horizon: int = 2,
        act_horizon: int = 16,
        img_size: int = 126,
        camera: str = "base_camera",
        max_episodes: int | None = None,
        success_only: bool = True,
        use_goal: bool = False,
        goal_slot: bool = False,
        task_goal: bool = False,
        feature_cache: str | Path | None = None,
        source: str = "motionplanning",
        clip_actions: bool = True,
        val_frac: float = 0.0,
        split: str = "train",
        split_seed: int = 0,
    ):
        self.h5_path = Path(h5_path)
        self.task = task
        self.instruction = TASK_INSTRUCTIONS[task]
        self.obs_horizon = obs_horizon
        self.act_horizon = act_horizon
        self.img_size = img_size
        self.camera = camera
        self.use_goal = use_goal
        # 演示里记录的是策略的**原始输出**，而环境执行前会把它裁到 [-1,1]。
        # 运动规划的演示本来就落在范围内（越界元素 0.0%），所以这是个空操作；
        # 但 PPO 的演示有 17.9% 的元素越界（夹爪最大到 ±4.85，均值 -2.005），
        # 不裁剪就会让"同一个动作"在两种来源里对应完全不同的数值 ——
        # 合并归一化之后运动规划的动作被压到 0.06 的尺度，模型分辨不出来，
        # 实测混合数据上 FM 和 BC 双双掉到 3%。
        self.clip_actions = clip_actions
        # 留出集的三个参数要能从 checkpoint 里读回来：评测脚本必须用**同一个**划分，
        # 否则"留出集误差"里混进了训练过的 episode，而这不会报错。
        self.val_frac = val_frac
        self.split = split
        self.split_seed = split_seed
        # goal_slot：所有任务都产出 qpos(9) + goal(3) + valid(1) = 13 维，
        # 演示里没有 goal_pos 的任务填零并把 valid 置 0。use_goal 那种"有就 12 维、
        # 没有就 9 维"的做法让 proprio_dim 在任务间跳动，多任务根本拼不起来。
        self.goal_slot = goal_slot
        # task_goal：每个任务从自己的 actor 取目标位置，proprio 恒为 12 维。
        # 与 use_goal 的观测空间完全一致，只是目标的来源按任务定义。
        self.task_goal = task_goal
        assert sum((use_goal, goal_slot, task_goal)) <= 1, \
            "use_goal / goal_slot / task_goal 三选一"
        self.goal_actor = TASK_GOAL_ACTOR.get(task) if task_goal else None
        self.has_goal = False

        # 冻结骨干的 patch token 缓存（scripts/build_feature_cache.py）。
        # 打开后 __getitem__ 返回的 "rgb" 不再是像素而是 (T, n_patches, 384) 的特征，
        # VisualEncoder 按维数自动分派。评测走环境实时图像，仍是像素路径。
        # 缓存目录名必须与 scripts/build_feature_cache.py 的命名一致，
        # 且要带上来源 —— 不同来源是不同的图像，共用一个目录会静默读错数据。
        _sfx = "" if source == "motionplanning" else f"_{source}"
        self.feature_cache = Path(feature_cache) / f"{task}{_sfx}_{camera}_{img_size}" \
            if feature_cache else None
        self._feat: np.ndarray | None = None
        if self.feature_cache is not None:
            meta = json.loads((self.feature_cache / "index.json").read_text())
            assert meta["img_size"] == img_size and meta["camera"] == camera, (
                f"缓存 {self.feature_cache} 是用 img_size={meta['img_size']} / "
                f"camera={meta['camera']} 建的，与当前请求不符")
            self.feat_index = meta["episodes"]
            self.feat_shape = (meta["n_patches"], meta["embed_dim"])
        self._file: h5py.File | None = None  # 每个 worker 各自延迟打开

        # ---- 只扫描元信息，建立索引 ----
        self.index: list[tuple[str, int]] = []
        self.episode_lengths: dict[str, int] = {}
        with h5py.File(self.h5_path, "r") as f:
            names = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))
            if max_episodes is not None:
                names = names[:max_episodes]
            kept, lengths = [], {}
            for name in names:
                ep = f[name]
                if success_only and not bool(np.asarray(ep["success"])[-1]):
                    continue
                T = ep["actions"].shape[0]
                if T < 1:
                    continue
                kept.append(name)
                lengths[name] = T
            # 切分放在筛选**之后**：先按 success_only / max_episodes 定下候选集，
            # 再从候选集里划留出。反过来（先切再筛）会让留出比例随失败轨迹的
            # 分布漂移，两个来源之间还对不齐。
            self.episode_names = split_episodes(kept, val_frac, split_seed, split)
            for name in self.episode_names:
                self.episode_lengths[name] = lengths[name]
                # start_t 可取 0..T-1：动作块不足 act_horizon 时在末尾重复最后一个动作
                self.index.extend((name, t) for t in range(lengths[name]))

            first = f[names[0]]
            self.act_dim = first["actions"].shape[-1]
            self.proprio_dim = first["obs"]["agent"]["qpos"].shape[-1]
            self.has_goal = "goal_pos" in first["obs"]["extra"]
            if use_goal:
                # StackCube / PegInsertion 的 obs 里没有 goal_pos（它们的目标是场景里
                # 的实物），所以 use_goal 只对 PickCube 这类带显式目标的任务成立。
                assert self.has_goal, (
                    f"{task} 的演示数据里没有 obs/extra/goal_pos，不能用 use_goal=true")
                self.proprio_dim += first["obs"]["extra"]["goal_pos"].shape[-1]
            elif goal_slot:
                self.proprio_dim += 4          # goal_pos(3) + valid(1)
            elif task_goal:
                assert self.goal_actor, f"{task} 没有登记目标 actor，见 TASK_GOAL_ACTOR"
                assert self.goal_actor in first["env_states"]["actors"], (
                    f"{task} 的演示里没有 env_states/actors/{self.goal_actor}")
                self.proprio_dim += 3

    def __len__(self) -> int:
        return len(self.index)

    @property
    def feat(self) -> np.ndarray:
        """每个 DataLoader worker 各自 mmap 一次。fork 出来的进程共享同一份页缓存。

        **不要加 MADV_RANDOM。** 三个任务的特征缓存合计 15.3GB，与整机内存相当，
        2026-09-13 凌晨内核为了留住这些页面回收了别的进程的匿名内存，把用户的
        编辑器杀掉了。当时试过用 MADV_RANDOM 关掉顺序预读来省内存，实测：

            可用内存   557-868MB  ->  5734MB
            训练吞吐   26.5 it/s  ->   6.0 it/s

        内存确实省下来了，但每次缺页都要单独去磁盘取，吞吐掉了四倍多，一轮
        60000 步从 38 分钟变成 1 小时 40 分钟。**这笔买卖不划算**，因为内存的
        主要改善来自把 DataLoader worker 从 3 个降到 2 个（少了一个 4GB 量级的
        进程），而不是来自关预读。所以保留 num_workers=2，去掉这个提示。

        真要再省内存，方向是减小缓存本身（当前 fp16、81 patch、384 维），
        或者给训练进程套一个内存 cgroup —— 后者在本机不可行，内存控制器在
        cgroup v1 上，用户级进程拿不到委派权限。
        """
        if self._feat is None:
            self._feat = np.load(self.feature_cache / "feat.f16", mmap_mode="r")
        return self._feat

    @property
    def file(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
        return self._file

    @staticmethod
    def _gather(dset, ts) -> np.ndarray:
        """按任意下标列表（可重复、可乱序）从 h5 dataset 取数据。

        h5py 的花式索引要求下标严格递增，而观测窗口在轨迹首尾会做 padding
        （重复第 0 帧 / 最后一个动作），必然产生重复下标。所以先取唯一值的
        递增子集，再用 inverse 映射还原成请求的顺序。
        """
        ts = np.asarray(ts)
        uniq, inv = np.unique(ts, return_inverse=True)
        return np.asarray(dset[uniq.tolist()])[inv]

    def _load_rgb(self, ep, ts) -> torch.Tensor:
        frames = self._gather(ep["obs"]["sensor_data"][self.camera]["rgb"], ts)
        return preprocess_obs_rgb(frames, self.img_size)

    def __getitem__(self, idx: int) -> dict:
        name, t = self.index[idx]
        ep = self.file[name]
        T = self.episode_lengths[name]

        # 观测窗口 [t-obs_horizon+1, t]，开头不足则重复第 0 帧（padding）
        obs_ts = [max(0, t - i) for i in reversed(range(self.obs_horizon))]
        if self.feature_cache is None:
            rgb = self._load_rgb(ep, obs_ts)
        else:
            # 按 episode 名字取偏移，不按序号 —— 序号错位是静默的，名字错位会 KeyError
            off, L = self.feat_index[name]
            assert L == T + 1 or L >= T, f"{name} 缓存帧数 {L} 与轨迹长度 {T} 不符"
            rgb = torch.from_numpy(
                np.asarray(self.feat[[off + i for i in obs_ts]], dtype=np.float32))
        if self.task_goal:
            goal = self._gather(ep["env_states"]["actors"][self.goal_actor], obs_ts)[:, :3]
        elif self.use_goal or (self.goal_slot and self.has_goal):
            goal = self._gather(ep["obs"]["extra"]["goal_pos"], obs_ts)
        else:
            goal = None
        proprio = build_proprio(
            self._gather(ep["obs"]["agent"]["qpos"], obs_ts), goal,
            goal_slot=self.goal_slot)

        # 动作块 [t, t+act_horizon)，末尾不足则重复最后一个动作
        act_ts = [min(T - 1, t + i) for i in range(self.act_horizon)]
        action = torch.from_numpy(self._gather(ep["actions"], act_ts)).float()
        if self.clip_actions:
            action = action.clamp(-1.0, 1.0)

        return {
            "rgb": rgb,
            "proprio": proprio,
            "action": action,
            "instruction": self.instruction,
        }

    def sample_actions(self, n: int = 20000) -> torch.Tensor:
        """随机采样一批动作，用于拟合归一化统计量（不必读全量）。"""
        rng = np.random.default_rng(0)
        picks = rng.choice(len(self.index), size=min(n, len(self.index)), replace=False)
        with h5py.File(self.h5_path, "r") as f:
            out = [f[self.index[i][0]]["actions"][self.index[i][1]] for i in picks]
        a = torch.from_numpy(np.stack(out)).float()
        return a.clamp(-1.0, 1.0) if self.clip_actions else a

    def __del__(self):
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass


def split_episodes(names: list[str], val_frac: float = 0.0, split_seed: int = 0,
                   split: str = "train") -> list[str]:
    """把 episode 名单切成训练集与留出集。

    **切在 episode 层面，不是样本层面。** 同一条轨迹相邻时刻的观测几乎一模一样
    （相机不动，机械臂挪了几毫米），按样本随机切的话，留出集里的每一帧在训练集里
    都有一个近乎重复的邻居，量出来的"泛化误差"其实还是拟合误差 —— 而且它会好看
    得多，于是这个错误不会被任何数字暴露出来。

    **划分只由 split_seed 决定，与训练的 seed 无关。** 多 seed 实验要量的是
    "换一个初始化，结论还成不成立"；如果每个 seed 的数据划分也跟着变，量出来的
    方差里就混进了"换一批数据"，多 seed 本来要回答的问题反而答不了。同理，
    不同方法之间做对照时三者必须拿到逐条相同的训练集，所以默认 split_seed=0
    写死在配置里，而不是继承 cfg.seed。

    val_frac=0 时留出集为空，训练集是全部 —— 这是本项目 2026-09-12 之前所有
    结果的配置，保留它是为了那些结果仍能原样复现。
    """
    assert split in ("train", "val", "all"), f"未知的 split: {split}"
    if split == "all":
        return list(names)
    if val_frac <= 0:
        return list(names) if split == "train" else []
    n_val = int(round(val_frac * len(names)))
    assert 0 < n_val < len(names), (
        f"val_frac={val_frac} 在 {len(names)} 条轨迹上切出 {n_val} 条留出，"
        f"不是一个可用的划分")
    perm = np.random.default_rng(split_seed).permutation(len(names))
    val = {names[i] for i in perm[:n_val]}
    # 保持原有顺序返回，便于人读日志时对得上 episode 编号
    return [n for n in names if (n in val) == (split == "val")]


def build_proprio(qpos, goal_pos=None, goal_slot: bool = False) -> torch.Tensor:
    """训练与评测共用的 proprio 组装。

    和 preprocess_obs_rgb 同理：两边各写一份是这类项目最隐蔽的 bug 来源。
    这里训练与评测不一致会撞出维度错误（9 vs 12）而不是静默跑偏，但拼接顺序
    错了就是静默的 —— 所以两边都从这一个函数走。评测侧的 use_goal 由
    run_eval.py 从 checkpoint 的 cfg 读取，自动跟随训练时的设置。

    qpos       (..., 9)   关节角 + 夹爪
    goal_pos   (..., 3)   目标位置；None 表示这个任务没有显式目标
    goal_slot  True 时**恒定**输出 13 维 = qpos(9) + goal(3) + valid(1)，
               没有目标就填零并把 valid 置 0。多任务必须走这条路：任务之间
               proprio_dim 一跳，ConcatDataset 和策略的输入层就对不上了。
               有效位不可省 —— 否则"目标在原点"与"没有目标"在网络看来一模一样。
    """
    q = torch.as_tensor(np.asarray(qpos)).float()
    if not goal_slot:
        if goal_pos is None:
            return q
        return torch.cat([q, torch.as_tensor(np.asarray(goal_pos)).float()], dim=-1)

    if goal_pos is None:
        g = torch.zeros(*q.shape[:-1], 3, dtype=q.dtype)
        valid = torch.zeros(*q.shape[:-1], 1, dtype=q.dtype)
    else:
        g = torch.as_tensor(np.asarray(goal_pos)).float()
        valid = torch.ones(*q.shape[:-1], 1, dtype=q.dtype)
    return torch.cat([q, g, valid], dim=-1)


def preprocess_obs_rgb(rgb_uint8, img_size: int = 126) -> torch.Tensor:
    """把环境返回的 RGB 观测处理成与训练数据完全一致的形式。

    训练和评测共用这一个函数 —— 两边各写一份预处理是这类项目最隐蔽的 bug 来源，
    归一化常数或通道顺序差一点，策略在仿真里就会莫名其妙地不工作。

    输入  (..., H, W, 3) uint8（tensor 或 ndarray）
    输出  (..., 3, img_size, img_size) float32，已按 ImageNet 统计量归一化
    """
    if not torch.is_tensor(rgb_uint8):
        rgb_uint8 = torch.from_numpy(np.asarray(rgb_uint8))
    x = rgb_uint8.float().div(255.0)
    lead = x.shape[:-3]
    x = x.reshape(-1, *x.shape[-3:]).permute(0, 3, 1, 2)      # (N,3,H,W)
    if x.shape[-1] != img_size:
        x = torch.nn.functional.interpolate(
            x, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean = torch.as_tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.as_tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return ((x - mean) / std).reshape(*lead, 3, img_size, img_size)


def collate(batch: list[dict]) -> dict:
    """默认 collate 会把字符串列表也堆叠，这里保持 instruction 为 list[str]。"""
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "proprio": torch.stack([b["proprio"] for b in batch]),
        "action": torch.stack([b["action"] for b in batch]),
        "instruction": [b["instruction"] for b in batch],
    }


def default_h5_path(task: str, demo_root: str | Path | None = None,
                    source: str = "motionplanning") -> Path:
    """`source` 选演示的来源目录。

    ManiSkill 对部分任务同时提供 `motionplanning` 和 `rl` 两套演示，它们是**两种
    不同的解法**：运动规划走的是规划出来的平滑轨迹，PPO 学出来的是另一套策略，
    PickCube 上中位轨迹长度分别是 74 步和 50 步。

    这件事对本项目是决定性的。单一来源的演示，实测条件动作分布是**宽的单峰**
    （三个任务的双峰系数 0.49/0.39/0.48，都低于正态的 0.555），
    于是"回归会塌到条件均值"这个选生成模型的理由不成立，等条件的 BC 反而更强。
    把两种来源混在一起才有真正的多模态 —— 那才是生成式动作头该赢的地方。
    见 scripts/conditional_spread.py 与 docs/debugging.md。
    """
    root = Path(demo_root or Path.home() / ".maniskill" / "demos")
    d = root / task / source
    # 后端后缀跟着**源轨迹**走，不跟着重放走：motionplanning 的演示是 CPU 后端录的，
    # rl 的是 GPU 后端录的，重放出来分别是 physx_cpu 和 physx_cuda。
    # 写死一个后缀会在换来源时炸出 FileNotFoundError。
    for backend in ("physx_cpu", "physx_cuda"):
        p = d / f"trajectory.rgb.{CONTROL_MODE}.{backend}.h5"
        if p.exists():
            return p
    return d / f"trajectory.rgb.{CONTROL_MODE}.physx_cpu.h5"


class MixedSourceDataset(Dataset):
    """把同一个任务的多份演示（不同解法）拼成一个数据集。

    存在的理由见 default_h5_path 的说明：单一来源的条件动作分布是单峰的，
    这时生成式动作头没有多模态可表示。混合来源是在**同一个任务上**制造多模态的
    最省事的办法，且不改任务、不改观测、不改评测协议 —— 只有数据的条件分布变了。
    """

    def __init__(self, task: str, sources: list[str], **kw):
        self.subsets = [ManiskillDataset(default_h5_path(task, source=s), task=task,
                                         source=s, **kw)
                        for s in sources]
        self.sources = sources
        self.task = task
        self.act_dim = self.subsets[0].act_dim
        self.proprio_dim = self.subsets[0].proprio_dim
        for d in self.subsets[1:]:
            assert (d.act_dim, d.proprio_dim) == (self.act_dim, self.proprio_dim)
        self.offsets = np.cumsum([0] + [len(d) for d in self.subsets])
        self.instruction = self.subsets[0].instruction
        # check_dataset 按这个属性决定用像素还是缓存特征的口径做检查。
        # 之前写死成 None，于是走缓存时它拿 ImageNet 的均值方差去检查
        # (2, 81, 384) 的特征，炸在 "rgb 应为 (T,3,H,W)"。
        self.feature_cache = self.subsets[0].feature_cache
        self.episode_lengths = {f"{s}:{k}": v for s, d in zip(sources, self.subsets)
                                for k, v in d.episode_lengths.items()}

    def __len__(self):
        return int(self.offsets[-1])

    def __getitem__(self, i):
        j = int(np.searchsorted(self.offsets, i, side="right") - 1)
        return self.subsets[j][i - int(self.offsets[j])]

    def sample_actions(self, n: int = 20000):
        return torch.cat([d.sample_actions(max(1, n // len(self.subsets)))
                          for d in self.subsets])


def build_multitask_dataset(tasks: list[str], **kw):
    """多任务：每个任务一个数据集，配等比例采样的 sampler（w1d5t1）。

    三个任务的轨迹长度不同（PickCube ~74 步，PegInsertion ~178 步），
    直接拼接会让长任务的样本数多出一倍多，模型会偏向它。
    这里按数据集大小的倒数加权，使三个任务被采到的期望次数相同。
    """
    from torch.utils.data import ConcatDataset, WeightedRandomSampler

    datasets = [ManiskillDataset(default_h5_path(t), task=t, **kw) for t in tasks]
    weights = torch.cat([
        torch.full((len(d),), 1.0 / len(d)) for d in datasets
    ])
    n_samples = min(len(d) for d in datasets) * len(datasets)
    sampler = WeightedRandomSampler(weights, num_samples=n_samples, replacement=True)
    return ConcatDataset(datasets), sampler, datasets
