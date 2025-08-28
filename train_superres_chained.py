

# =========================
# PATHS (EDIT THESE)
# =========================
LR_DIR  = r"E:/25 Aug 2025_project/converted_particles_Low"
HR_DIR  = r"E:/25 Aug 2025_project/converted_particles_High"
OUT_DIR = r"E:/25 Aug 2025_project/checkpoints"

# =========================
# TRAINING / MODEL PARAMS
# =========================
EPOCHS       = 10       # epochs (fast because we use patches)
BATCHES      = 200      # batches per epoch (random patches each time)
PATCH_LR     = 1000     # LR points per patch
PATCH_HR     = 8000     # HR points per patch
K            = 8        # children per LR parent (keep small for speed)
LRATE        = 1e-3
DEVICE       = "cuda"   # "cuda" or "cpu"; will auto-fallback to CPU if CUDA not available

# Graph / GNN caps
MAX_DEGREE   = 12       # max neighbors per node in radius graph
EDGE_CHUNK   = 50_000   # edges processed per chunk (chunked message passing)
NUM_REFINES  = 2        # refinement steps
HIDDEN       = 64       # model width (64 is fast; 128 for more capacity)
W_POS, W_VEL = 1.0, 0.2 # loss weights (offset vs velocity)

# Visualization
VIS_EVERY_BATCH = 20    # save a PNG every N batches
VIS_POINT_LIMIT = 20_000
SAVE_PATCH_CSV  = False # also save predicted patch as CSV (off by default)

# =========================
# IMPORTS
# =========================
import os, re, glob, random, math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# UTILITIES
# =========================
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    vis_dir = os.path.join(OUT_DIR, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    return vis_dir

def frame_index(path: str) -> int:
    m = re.search(r'(\d+)(?=\.csv$)', os.path.basename(path))
    return int(m.group(1)) if m else -1

def pair_lr_hr(lr_dir, hr_dir):
    lr_map = {frame_index(p): p for p in glob.glob(os.path.join(lr_dir, "*.csv"))}
    hr_map = {frame_index(p): p for p in glob.glob(os.path.join(hr_dir, "*.csv"))}
    common = sorted(set(lr_map) & set(hr_map))
    if not common:
        raise RuntimeError("No paired LR/HR frames found. Check directories/filenames.")
    return [lr_map[i] for i in common], [hr_map[i] for i in common]

def load_csv(path):
    df = pd.read_csv(path)
    req = {"x","y","z","vx","vy","vz","type"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"{path} missing columns: {sorted(miss)}")
    return df

def build_graph(pos, radius=0.5, max_degree=12, knn=6):
    """
    Memory-safe graph:
      - per-node radius neighbors with degree cap
      - kNN fallback to avoid isolates
    Returns: edge_index [2,E], edge_attr [E,4] = (dx,dy,dz,dist)
    """
    N = len(pos)
    if N == 0:
        return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,4))
    tree = cKDTree(pos)
    edges = []
    for i in range(N):
        nb = tree.query_ball_point(pos[i], r=radius)
        nb = [j for j in nb if j != i]
        if not nb:
            continue
        if len(nb) > max_degree:
            cand = np.asarray(nb)
            d = np.linalg.norm(pos[cand] - pos[i], axis=1)
            nb = cand[np.argsort(d)[:max_degree]]
        for j in nb:
            edges.append((i, j))
    if N > 1:
        _, nbrs = tree.query(pos, k=min(knn+1, N))
        for i, row in enumerate(nbrs[:, 1:]):
            for j in row:
                edges.append((i, j))
    if not edges:
        return torch.zeros((2,0), dtype=torch.long), torch.zeros((0,4))
    edges = np.unique(np.array(edges), axis=0)
    src, dst = edges[:,0], edges[:,1]
    rel = pos[dst] - pos[src]
    dist = np.linalg.norm(rel, axis=1, keepdims=True)
    ea = np.hstack([rel, dist]).astype(np.float32)
    return torch.from_numpy(edges.T).long(), torch.from_numpy(ea).float()

def assign_hr_to_lr_parents(posL, posH, K):
    """Nearest LR parent for each HR; per LR keep up to K closest children."""
    tree = cKDTree(posL)
    _, p = tree.query(posH, k=1)
    groups = [[] for _ in range(len(posL))]
    for j, i in enumerate(p):
        groups[i].append(j)
    out = []
    for i, g in enumerate(groups):
        if not g:
            out.append([])
            continue
        d = np.linalg.norm(posH[g] - posL[i], axis=1)
        order = np.argsort(d)
        out.append([g[k] for k in order[:K]])
    return out

# =========================
# MODELS
# =========================
class MLP(nn.Module):
    def __init__(self, in_ch, out_ch, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_ch, hid), nn.ReLU(),
            nn.Linear(hid, hid),   nn.ReLU(),
            nn.Linear(hid, out_ch)
        )
    def forward(self, x): return self.net(x)

class GNN(nn.Module):
    def __init__(self, in_ch=6, edge_in=4, hid=64, mp=2):
        super().__init__()
        self.enc  = MLP(in_ch, hid, hid)
        self.edge = MLP(hid + edge_in, hid, hid)
        self.upd  = MLP(hid + hid, hid, hid)
        self.mp   = mp
    def forward(self, x, ei, ea):
        h = self.enc(x)         # [N,H]
        src, dst = ei
        for _ in range(self.mp):
            agg = torch.zeros_like(h)
            E = src.numel()
            for s in range(0, E, EDGE_CHUNK):
                e = min(E, s + EDGE_CHUNK)
                m = self.edge(torch.cat([h[src[s:e]], ea[s:e]], dim=1))
                agg.index_add_(0, dst[s:e], m)
            h = h + self.upd(torch.cat([h, agg], dim=1))
        return h

class Upsampler(nn.Module):
    """Per LR node → K children: predicts [Δpos(3), vel(3)] in normalized patch coords."""
    def __init__(self, in_ch, K, hid=64):
        super().__init__()
        self.K = K
        self.mlp = MLP(in_ch, 6 * K, hid)
    def forward(self, h):
        y = self.mlp(h)
        return y.view(h.shape[0] * self.K, 6)

class Refiner(GNN):
    def __init__(self, hid=64, mp=2):
        super().__init__(in_ch=12, edge_in=4, hid=hid, mp=mp)
        self.head = MLP(hid, 6, hid)
    def forward(self, x, ei, ea):
        h = super().forward(x, ei, ea)
        return self.head(h)

# =========================
# VISUALIZATION
# =========================
def _thin(arr, nmax):
    if len(arr) <= nmax: return arr
    idx = np.random.choice(len(arr), size=nmax, replace=False)
    return arr[idx]

def save_patch_visual(vis_dir, epoch, batch_i, posL, posH, posPred, title_suffix=""):
    # thin for speed/clarity
    posL2  = _thin(posL, min(5_000, VIS_POINT_LIMIT))
    posH2  = _thin(posH, VIS_POINT_LIMIT)
    posPr2 = _thin(posPred, VIS_POINT_LIMIT)

    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(posH2[:,0],  posH2[:,1],  s=1, c="tab:green", alpha=0.25, label="HR (GT)")
    ax.scatter(posPr2[:,0], posPr2[:,1], s=1, c="tab:red",   alpha=0.35, label="HR (Pred)")
    ax.scatter(posL2[:,0],  posL2[:,1],  s=6, c="tab:blue",  alpha=0.80, label="LR")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Epoch {epoch}  Batch {batch_i}  {title_suffix}")
    ax.legend(loc="upper right", markerscale=6, frameon=True)
    ax.grid(True, ls="--", alpha=0.3)
    out_path = os.path.join(vis_dir, f"epoch_{epoch:03d}_batch_{batch_i:04d}.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def save_loss_curve(vis_dir, loss_hist):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(loss_hist, lw=2)
    ax.set_xlabel("Batch (global)")
    ax.set_ylabel("Train loss")
    ax.grid(True, ls="--", alpha=0.4)
    fig.tight_layout()
    out_path = os.path.join(vis_dir, "loss_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

# =========================
# TRAIN
# =========================
def main():
    print(">>> START train_fast_superres_vis.py", flush=True)

    # sanity checks
    assert os.path.isdir(LR_DIR), f"LR_DIR not found: {LR_DIR}"
    assert os.path.isdir(HR_DIR), f"HR_DIR not found: {HR_DIR}"
    vis_dir = ensure_dirs()

    lr_files = glob.glob(os.path.join(LR_DIR, "*.csv"))
    hr_files = glob.glob(os.path.join(HR_DIR, "*.csv"))
    print(f"LR csvs: {len(lr_files)} | HR csvs: {len(hr_files)}", flush=True)

    lr_files, hr_files = pair_lr_hr(LR_DIR, HR_DIR)
    print(f"Found {len(lr_files)} paired frames", flush=True)

    # device
    dev = DEVICE if (DEVICE == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {dev}", flush=True)

    # models + optimizer
    gnn = GNN(in_ch=6, edge_in=4, hid=HIDDEN, mp=2).to(dev)
    up  = Upsampler(in_ch=HIDDEN, K=K, hid=HIDDEN).to(dev)
    ref = Refiner(hid=HIDDEN, mp=2).to(dev)
    opt = torch.optim.Adam(
        list(gnn.parameters()) + list(up.parameters()) + list(ref.parameters()),
        lr=LRATE
    )

    global_loss_hist = []
    global_batch_idx = 0

    for epoch in range(1, EPOCHS+1):
        total = 0.0
        pbar = tqdm(range(BATCHES), desc=f"Epoch {epoch}")

        for bi in pbar:
            # pick a random paired frame + sample patches
            idx = random.randrange(len(lr_files))
            dl, dh = load_csv(lr_files[idx]), load_csv(hr_files[idx])

            dl = dl.sample(min(PATCH_LR, len(dl)), replace=False)
            dh = dh.sample(min(PATCH_HR, len(dh)), replace=False)

            posL = dl[["x","y","z"]].to_numpy().astype(np.float32)
            velL = dl[["vx","vy","vz"]].to_numpy().astype(np.float32)
            posH = dh[["x","y","z"]].to_numpy().astype(np.float32)
            velH = dh[["vx","vy","vz"]].to_numpy().astype(np.float32)

            # per-patch normalization
            mean = posL.mean(0)
            std  = posL.std(0) + 1e-6
            posL_n = (posL - mean) / std
            posH_n = (posH - mean) / std

            # LR graph
            ei_LR, ea_LR = build_graph(posL_n, radius=0.5, max_degree=MAX_DEGREE, knn=6)
            x_LR = torch.from_numpy(np.hstack([posL_n, velL])).float().to(dev)
            ei_LR, ea_LR = ei_LR.to(dev), ea_LR.to(dev)

            # encode + initial upsample
            h    = gnn(x_LR, ei_LR, ea_LR)     # [N_lr, H]
            init = up(h)                        # [N_lr*K, 6]
            parent = torch.from_numpy(np.repeat(posL_n, K, 0)).float().to(dev)
            dpos   = init[:, :3]                # normalized offsets
            vch    = init[:, 3:6]

            # targets via nearest parent assignment
            groups = assign_hr_to_lr_parents(posL, posH, K)
            tgt = []
            for pi, g in enumerate(groups):
                if not g:
                    tgt.append(np.zeros((K,6), np.float32)); continue
                ids   = g[:K]
                dposH = posH_n[ids] - posL_n[pi]
                feat  = np.hstack([dposH, velH[ids]])
                if len(ids) < K:
                    feat = np.vstack([feat, np.zeros((K-len(ids),6), np.float32)])
                tgt.append(feat)
            target = torch.from_numpy(np.vstack(tgt)).float().to(dev)

            # chained refinement
            for _ in range(NUM_REFINES):
                child  = parent + dpos
                ei, ea = build_graph(child.detach().cpu().numpy(), radius=0.35, max_degree=MAX_DEGREE, knn=6)
                ei, ea = ei.to(dev), ea.to(dev)
                x_child = torch.cat([child, parent, dpos, vch], 1)  # [*,12]
                res = ref(x_child, ei, ea)                         # [*,6] residuals
                dpos = dpos + res[:, :3]
                vch  = vch  + res[:, 3:6]

            # loss + step
            loss = W_POS * F.mse_loss(dpos, target[:, :3]) + W_VEL * F.mse_loss(vch, target[:, 3:6])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(gnn.parameters())+list(up.parameters())+list(ref.parameters()), 1.0)
            opt.step()

            total += loss.item()
            global_loss_hist.append(loss.item())
            global_batch_idx += 1
            pbar.set_postfix(loss=f"{total/(bi+1):.4f}")

            # visualization every N batches
            if (bi % VIS_EVERY_BATCH) == 0:
                with torch.no_grad():
                    child_world = (child.cpu().numpy() * std + mean)
                    posL_world  = (posL_n * std + mean)
                    posH_world  = (posH_n * std + mean)
                    png = save_patch_visual(vis_dir, epoch, bi, posL_world, posH_world, child_world, title_suffix=f"K={K}")
                    if SAVE_PATCH_CSV:
                        csvp = os.path.join(vis_dir, f"patch_epoch_{epoch:03d}_batch_{bi:04d}.csv")
                        pd.DataFrame({"x":child_world[:,0],"y":child_world[:,1],"z":child_world[:,2]}).to_csv(csvp, index=False)

        save_loss_curve(vis_dir, global_loss_hist)
        print(f"Epoch {epoch:02d} avg loss = {total/BATCHES:.4f}", flush=True)

    # save model
    ckpt_path = os.path.join(OUT_DIR, "fast_superres.pt")
    torch.save({
        "gnn": gnn.state_dict(),
        "up":  up.state_dict(),
        "ref": ref.state_dict(),
        "config": {
            "K": K, "hidden": HIDDEN, "num_refines": NUM_REFINES,
            "max_degree": MAX_DEGREE, "edge_chunk": EDGE_CHUNK
        }
    }, ckpt_path)
    print(f"✅ Model saved to: {ckpt_path}")
    print(f"🖼️ Visualizations saved to: {os.path.join(OUT_DIR, 'vis')}")

# =========================
# MAIN GUARD (do not remove)
# =========================
if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
