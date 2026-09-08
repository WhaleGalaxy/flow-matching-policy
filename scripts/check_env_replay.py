"""把演示动作原样回放进评测环境，分离"环境配置问题"与"策略问题"。

策略成功率为 0 时，第一件事是确认评测环境本身是对的 —— 否则会去调模型，
而问题在环境。用记录的 seed 复现初始状态、把演示动作原样喂进去，成功率应接近
100%；达不到就说明 make_env 的控制模式、观测或成功判据与演示不一致
（docs/debugging.md 里第一轮排查就栽在控制模式不匹配上）。

    python scripts/check_env_replay.py
"""
import sys, json, h5py, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import make_env
from src.data.dataset import default_h5_path

N = 10
for task in ["PickCube-v1", "StackCube-v1", "PegInsertionSide-v1"]:
    h5p = default_h5_path(task); meta = json.load(open(str(h5p).replace(".h5", ".json")))
    eps = {e["episode_id"]: e for e in meta["episodes"]}
    env = make_env(task)
    ok, lens = 0, []
    with h5py.File(h5p, "r") as f:
        names = sorted(f.keys(), key=lambda s: int(s.split("_")[1]))[:N]
        for name in names:
            eid = int(name.split("_")[1]); rk = eps[eid]["reset_kwargs"]
            env.reset(seed=rk.get("seed"), options=rk.get("options", {}))
            acts = np.asarray(f[name]["actions"])
            succ = False
            for a in acts:
                _, _, term, trunc, info = env.step(a)
                s = info["success"]
                succ = bool(s.cpu().reshape(-1)[0]) if hasattr(s, "cpu") else bool(np.asarray(s).reshape(-1)[0])
                if succ: break
            ok += succ; lens.append(len(acts))
    env.close()
    print(f"{task:22s} 回放成功 {ok}/{N}   演示平均长度 {np.mean(lens):.0f} 步")
