"""Paper-style figures for the multi-chain task (van Hasselt et al. 2021, Figs. 2 and 3).

Produces (in --outdir):
  fig3_multichain.png         the environment diagram, as Fig. 3
  fig2_tabular_heatmaps.png   Fig. 2 style: rows = learners, columns = episodes; each panel is the task as an image
                              (rows = chains, columns = start | chain positions | shared state), colour = value
  fig3_tabular_values.png     Fig. 3 diagram with every state coloured by its value, rows = learners, columns = episodes
  fig2_deep_heatmaps.png / fig3_deep_values.png   the same for deep agents (if --deep files are given)
  tabular_data.npz + tabular_values.csv           all simulated values (every learner, every snapshot, every state)

Tabular learners (one-hot features, step sizes alpha = beta = sqrt(1/k), k = episode number, as in the paper):
  TD(0) | TD(lambda=0.9) | ET(lambda=0.9) z applied now (paper Alg. 1) | ET(lambda=0.9) z one step late (Alg. 3 timing)

Environment: start -> one of m chains (random) of length n -> shared state -> terminal.
Reward 0 everywhere except the last transition: +1 w.p. 0.9 and -1 w.p. 0.1 (paper) [--reward paper],
or 1 + sigma * N(0,1) [--reward gauss, matching the deep R30/R32 runs]. gamma = 1 (paper) or 0.99 (deep runs).

Usage:
  python make_paper_figures.py --m 8 --n 4 --outdir figs
  python make_paper_figures.py --m 8 --reward gauss --gamma 0.99 --deep "QRC (λ=0)=path/value_snapshots.npz" \
                               "QRC-ET (no lag)=path/value_snapshots.npz" --outdir figs
"""
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11})
CMAP = matplotlib.colormaps["magma"].copy()
CMAP.set_bad("black")


# ------------------------------------------------------------------------------------------- environment / learners
BERN_P = 0.2                                                    # success probability for --reward bern


def episode(rng, m, n, reward, sigma):
    c = int(rng.integers(m))
    states = [0] + [1 + c * n + i for i in range(n)] + [1 + m * n]
    if reward == "paper":
        r_end = 1.0 if rng.random() < 0.9 else -1.0
    elif reward == "bern":                                       # sparse: +1 w.p. p, else 0 (paper Fig. 1/2 reward)
        r_end = 1.0 if rng.random() < BERN_P else 0.0
    else:
        r_end = 1.0 + sigma * float(rng.standard_normal())
    return states, [0.0] * (len(states) - 1) + [r_end]


def true_values(m, n, reward, gamma):
    mean = 0.8 if reward == "paper" else BERN_P if reward == "bern" else 1.0
    S = 1 + m * n + 1
    v = np.zeros(S)
    v[S - 1] = mean
    for c in range(m):
        for i in range(n):
            v[1 + c * n + i] = gamma ** (n - i) * mean
    v[0] = gamma ** (n + 1) * mean
    return v


def simulate(method, m, n, episodes, snap_eps, reward, sigma, gamma, lam, seed):
    rng = np.random.default_rng(seed)
    S = 1 + m * n + 1
    w, Z, cnt = np.zeros(S), np.zeros((S, S)), np.zeros(S)
    snaps = {}
    for k in range(1, episodes + 1):
        a = b = np.sqrt(1.0 / k)
        states, rewards = episode(rng, m, n, reward, sigma)
        e, z_prev = np.zeros(S), np.zeros(S)
        for t, s in enumerate(states):
            x = np.zeros(S); x[s] = 1.0
            v_next = w[states[t + 1]] if t + 1 < len(states) else 0.0
            delta = rewards[t] + gamma * v_next - w[s]
            e = gamma * lam * e + x
            cnt[s] += 1
            Z[:, s] += b * (e - Z[:, s])
            if method == "TD(0)" or method.startswith("TD("):
                u = e
            elif method.endswith("now"):
                u = Z[:, s]                                    # Alg. 1: expected trace of the current state
            else:
                u = gamma * lam * z_prev + x                   # Alg. 3 timing (eta = 0): z reaches the update one step late
            w += a * delta * u
            z_prev = Z[:, s].copy()
        if k in snap_eps:
            snaps[k] = w.copy()
    return snaps


# ------------------------------------------------------------------------------------------- drawing helpers
def as_image(values, m, n):
    """values (S,) -> image (m, n+2): start and shared state in the middle row, chains in between, black elsewhere."""
    img = np.full((m, n + 2), np.nan)
    mid = m // 2
    img[mid, 0] = values[0]
    for c in range(m):
        img[c, 1:n + 1] = values[1 + c * n: 1 + (c + 1) * n]
    img[mid, n + 1] = values[-1]
    return img


def draw_diagram(ax, colors, m, n, show_chains=None, node=0.32, arrow_color="0.3", fan_arrows=True):
    """Fig-3-style diagram. colors: list of S facecolors. show_chains: chain indices to draw (others as dots)."""
    show = list(range(m)) if show_chains is None else show_chains
    gap = 1.0
    ys = {c: (len(show) - 1 - j) * gap for j, c in enumerate(show)}
    mid = (len(show) - 1) * gap / 2
    xs = lambda i: 1.4 + i * 1.0
    start, shared = (0.0, mid), (xs(n - 1) + 1.4, mid)

    def circ(xy, fc):
        ax.add_patch(plt.Circle(xy, node, facecolor=fc, edgecolor="0.25", lw=1.0, zorder=3))

    def arrow(p, q, rad=0.0):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=arrow_color,
                                     shrinkA=node * 72 / 2.2, shrinkB=node * 72 / 2.2,
                                     connectionstyle=f"arc3,rad={rad}", zorder=2))
    circ(start, colors[0])
    for c in show:
        y = ys[c]
        for i in range(n):
            circ((xs(i), y), colors[1 + c * n + i])
            if i:
                arrow((xs(i - 1), y), (xs(i), y))
        if fan_arrows:
            arrow(start, (xs(0), y), rad=0.25 * np.sign(y - mid) if y != mid else 0.0)
            arrow((xs(n - 1), y), shared, rad=-0.25 * np.sign(y - mid) if y != mid else 0.0)
        else:
            ax.plot([start[0], xs(0)], [start[1], y], color="0.75", lw=0.6, zorder=1)
            ax.plot([xs(n - 1), shared[0]], [y, shared[1]], color="0.75", lw=0.6, zorder=1)
    if show_chains is not None and len(show) < m:                # dotted markers for hidden chains
        for i in range(n):
            for j in range(len(show) - 1):
                for d in (0.35, 0.5, 0.65):
                    ax.plot(xs(i), ys[show[j]] - d * gap, "o", ms=2.2, color="0.75", zorder=1)
    circ(shared, colors[-1])
    ax.add_patch(Rectangle((shared[0] + 0.85, mid - 0.2), 0.4, 0.4, color="black", zorder=3))
    arrow(shared, (shared[0] + 0.85, mid))
    ax.set_xlim(-0.6, shared[0] + 1.6); ax.set_ylim(-0.7, (len(show) - 1) * gap + 0.7)
    ax.set_aspect("equal"); ax.axis("off")


# ------------------------------------------------------------------------------------------- figures
def fig3_environment(m, n, out):
    S = 1 + m * n + 1
    blues = matplotlib.colormaps["Blues"]
    colors = ["white"] + [blues(0.25 + 0.5 * c / max(m - 1, 1)) for c in range(m) for _ in range(n)] + ["#ff8c1a"]
    fig, ax = plt.subplots(figsize=(6.5, 2.6))
    draw_diagram(ax, colors, m, n, show_chains=[0, m // 2, m - 1])
    ax.set_title(f"Multi-chain environment (m = {m} chains of length n = {n})", fontsize=11)
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", out)


def fig2_heatmaps(rows, cols, col_titles, m, n, vmax, out, title):
    """rows: [(name, {col: values})]. Paper Fig. 2 layout."""
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.75 * len(cols) + 0.6, 1.75 * len(rows) + 0.5), squeeze=False)
    for r, (name, snaps) in enumerate(rows):
        for c, key in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(as_image(snaps[key], m, n), cmap=CMAP, vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_facecolor("black")
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                ax.set_ylabel(name, fontsize=10)
    over = any(np.nanmax(v) > vmax for _, sn in rows for v in sn.values())
    fig.suptitle(title, fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 0.93, 1))
    cax = fig.add_axes([0.945, 0.15, 0.015, 0.7])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap=CMAP), cax=cax,
                 extend="max" if over else "neither", label="value estimate")
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", out)


def fig3_values(rows, cols, col_titles, m, n, vmax, out, title):
    pw = 2.6                                                     # panel width (inches); height follows the drawing
    ph = pw * ((m - 1) + 1.4) / ((1.4 + (n - 1) + 1.4) + 2.2)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(pw * len(cols) + 1.6, ph * len(rows) + 0.7), squeeze=False,
                             gridspec_kw={"wspace": 0.05, "hspace": 0.12})
    norm = plt.Normalize(0, vmax)
    for r, (name, snaps) in enumerate(rows):
        for c, key in enumerate(cols):
            cols_ = [CMAP(norm(max(v, 0))) for v in snaps[key]]
            draw_diagram(axes[r, c], cols_, m, n, node=0.36, fan_arrows=False)
            if r == 0:
                axes[r, c].set_title(col_titles[c], fontsize=10)
        axes[r, 0].text(-0.05, 0.5, name, transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=10)
    over = any(np.nanmax(v) > vmax for _, sn in rows for v in sn.values())
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=CMAP), ax=axes, shrink=0.6, pad=0.01, label="value estimate",
                 extend="max" if over else "neither")
    top = axes[0, 0].get_position().y1
    fig.suptitle(title, fontsize=11, y=min(0.995, top + 0.035))
    fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig); print("saved", out)


def trajectory_figure(a, out):
    """Mean value over all states vs episode for every deep run (+ tabular TD/ET on the same task), and the reward scale."""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.3), gridspec_kw={"width_ratios": [2.2, 1]})
    dtrue = None
    for spec in a.deep:
        name, path = spec.rsplit("=", 1)
        d = np.load(path)
        if "episode" not in d.files or not np.any(d["episode"] > 0):
            continue
        dtrue, dm, dn = d["true"], int(d["chains"]), int(d["chain_len"])
        ax.plot(d["episode"], d["values"].mean(1), marker="o", ms=2.5, lw=1.2, label=name.replace("\\n", " ").replace("\n", " "))
        if "reward_scale" in d.files:
            ax2.plot(d["episode"], d["reward_scale"], lw=1.2, label=name.replace("\\n", " ").replace("\n", " "))
    if dtrue is None:
        plt.close(fig); return
    eps = sorted(set(np.unique(np.concatenate([np.load(sp.rsplit("=", 1)[1])["episode"] for sp in a.deep]))) - {0})
    for meth, lam, label in [(f"TD(λ={a.lam})", a.lam, f"Tabular TD(λ = {a.lam})"), (f"ET(λ={a.lam}) now", a.lam, f"Tabular ET(λ = {a.lam})")]:
        ts = simulate(meth, dm, dn, max(eps), set(eps), a.reward if a.reward == "bern" else "gauss", a.sigma, 0.99, lam, a.seed)
        ax.plot(eps, [ts[k].mean() for k in eps], ls="--", lw=1.4, label=label)
    ax.axhline(dtrue.mean(), color="black", ls=":", lw=1.2, label="true value")
    ax.set_xscale("log"); ax.set_xlabel("episode"); ax.set_ylabel("mean value estimate over all states")
    ax.set_title("Value estimates over training (one seed)"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
    ax2.set_xscale("log"); ax2.set_xlabel("episode"); ax2.set_ylabel("reward scale (agent's normaliser)")
    ax2.set_title("Reward normalisation scale"); ax2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=180, bbox_inches="tight"); plt.close(fig); print("saved", out)


def reward_marks(m, n, reward, sigma, seed, targets):
    """Episode numbers at which the k-th positive reward arrives (same random stream as simulate and the env)."""
    rng, count, marks = np.random.default_rng(seed), 0, {}
    for ep in range(1, 10 ** 6):
        _, rw = episode(rng, m, n, reward, sigma)
        count += rw[-1] > 0
        for t in targets:
            if count >= t and t not in marks:
                marks[t] = ep
        if len(marks) == len(targets):
            return [marks[t] for t in targets]


def reward_title(k, ep):
    word = {1: "first reward", 2: "second reward"}.get(k, f"~{k:,} rewards")
    return f"Episode {ep:,}\n{word}"


def env_text(a, m, n):
    if a.reward == "paper":
        return f"{m} chains, n = {n}, reward +1 (p=0.9) / −1 (p=0.1)"
    if a.reward == "bern":
        return f"{m} chains, n = {n}, reward +1 with p = {a.p}, else 0"
    return f"{m} chains, n = {n}, reward 1 + {a.sigma}·N(0,1)"


def scale_top(a, true, rows):
    if a.vmax:
        return a.vmax
    peak = max(np.nanmax(v) for _, sn in rows for v in sn.values())
    return float(2 * true.max()) if peak > 1.05 * true.max() else float(true.max())


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m", type=int, default=8); ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--episodes", nargs="+", type=int, default=[1, 3, 10, 100, 1000])
    ap.add_argument("--reward", choices=["paper", "gauss", "bern"], default="paper"); ap.add_argument("--p", type=float, default=0.2)
    ap.add_argument("--by-rewards", nargs="+", type=int, default=None, help="columns at the k-th reward, e.g. 1 2 20 200 2000"); ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=1.0); ap.add_argument("--lam", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deep", nargs="*", default=[], help='"name=path/value_snapshots.npz" (from --snapshot_every)')
    ap.add_argument("--deep-steps", nargs="+", type=int, default=[2000, 6000, 20000, 50000, 98000])
    ap.add_argument("--deep-episodes", nargs="+", type=int, default=[1, 3, 10, 100, 1000],
                    help="used when the deep files are value_snapshots_episodes.npz")
    ap.add_argument("--with-tabular", action="store_true", help="add tabular TD(lambda) and ET rows (same task) above the deep rows")
    ap.add_argument("--outdir", default="figs")
    ap.add_argument("--vmax", type=float, default=None, help="top of the colour scale (default: deep = largest value, rounded up)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    m, n = a.m, a.n
    true = true_values(m, n, a.reward, a.gamma)
    vmax = float(true.max())

    fig3_environment(m, n, os.path.join(a.outdir, "fig3_multichain.png"))

    methods = [("TD(0)", 0.0), (f"TD(λ={a.lam})", a.lam), (f"ET(λ={a.lam}) now", a.lam), (f"ET(λ={a.lam}) late", a.lam)]
    names = {"TD(0)": "TD(0)", f"TD(λ={a.lam})": f"TD(λ = {a.lam})", f"ET(λ={a.lam}) now": f"ET(λ = {a.lam})\nz applied now",
             f"ET(λ={a.lam}) late": f"ET(λ = {a.lam})\nz one step late"}
    global BERN_P
    BERN_P = a.p
    true = true_values(m, n, a.reward, a.gamma)
    if a.by_rewards:
        eps = reward_marks(m, n, a.reward, a.sigma, a.seed, a.by_rewards)
        titles = [reward_title(k, e) for k, e in zip(a.by_rewards, eps)]
    else:
        eps = sorted(set(a.episodes))
        titles = [f"Episode {k:,}" for k in eps]
    rows, records = [], []
    for meth, lam in methods:
        snaps = simulate(meth, m, n, max(eps), set(eps), a.reward, a.sigma, a.gamma, lam, a.seed)
        rows.append((names[meth], snaps))
        for k, v in snaps.items():
            records.append((meth, k, *v))
    rows.append(("True values", {k: true for k in eps}))
    env_txt = env_text(a, m, n)
    vmax = scale_top(a, true, rows)
    fig2_heatmaps(rows, eps, titles, m, n, vmax, os.path.join(a.outdir, "fig2_tabular_heatmaps.png"),
                  f"Tabular value estimates over episodes | {env_txt}")
    fig3_values(rows, eps, titles, m, n, vmax, os.path.join(a.outdir, "fig3_tabular_values.png"),
                f"Tabular value estimates on the multi-chain | {env_txt}")
    np.savez_compressed(os.path.join(a.outdir, "tabular_data.npz"), true=true, episodes=np.array(eps),
                        **{meth: np.array([dict(rows[i][1])[k] for k in eps]) for i, (meth, _) in enumerate(methods)})
    with open(os.path.join(a.outdir, "tabular_values.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["learner", "episode"] + [f"state_{i}" for i in range(len(true))])
        wr.writerow(["true", ""] + list(true))
        wr.writerows(records)
    print("saved tabular_data.npz and tabular_values.csv")

    if a.deep:
        drows, by_episode = [], False
        for spec in a.deep:
            name, path = spec.rsplit("=", 1)
            name = name.replace("\\n", "\n")
            d = np.load(path)
            dm, dn, dtrue = int(d["chains"]), int(d["chain_len"]), d["true"]
            by_episode = "episode" in d.files and np.any(d["episode"] > 0)
            keys = a.deep_episodes if by_episode else a.deep_steps
            axis = d["episode"] if by_episode else d["step"]
            snaps = {}
            for w in (eps if a.by_rewards and by_episode else keys):
                i = max(0, int(np.searchsorted(axis, w, side="right")) - 1)
                snaps[w] = d["values"][i]
            drows.append((name, snaps))
        keys = a.deep_episodes if by_episode else a.deep_steps
        if a.by_rewards and by_episode:
            keys = eps
        if a.with_tabular and by_episode:                       # same task, same episodes, tabular reference rows
            treward, tgamma = (a.reward if a.reward == "bern" else "gauss", 0.99)
            for meth, lam, label in [(f"TD(λ={a.lam})", a.lam, f"Tabular TD(λ = {a.lam})"),
                                     (f"ET(λ={a.lam}) now", a.lam, f"Tabular ET(λ = {a.lam})")]:
                ts = simulate(meth, dm, dn, max(keys), set(keys), treward, a.sigma, tgamma, lam, a.seed)
                drows.insert(len([r for r in drows if r[0].startswith("Tabular")]), (label, ts))
        drows.append(("True values", {w: dtrue for w in keys}))
        dtitles = titles if (a.by_rewards and by_episode) else [f"Episode {w:,}" for w in keys] if by_episode else [f"Step {w:,}\n(≈ episode {w // (dn + 2):,})" for w in keys]
        vmax = scale_top(a, dtrue, drows)                        # true value sits mid-scale; arrow marks values above
        trajectory_figure(a, os.path.join(a.outdir, "fig_value_over_episodes.png"))
        fig2_heatmaps(drows, keys, dtitles, dm, dn, vmax, os.path.join(a.outdir, "fig2_deep_heatmaps.png"),
                      f"QRC vs QRC-ET: value estimates over training | {env_text(a, dm, dn)}")
        if dm <= 16:
            fig3_values(drows, keys, dtitles, dm, dn, vmax, os.path.join(a.outdir, "fig3_deep_values.png"),
                        f"QRC vs QRC-ET on the multi-chain | {env_text(a, dm, dn)}")


if __name__ == "__main__":
    main()
