# %% [markdown]
# ## Voronoi + Delaunay example — 256 × 256 grid, 256 random seeds

# %% — imports and computation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from delauney import RegularDelaunay, GridTriangulation


# %% — barycentric lambda heatmap diagnostic
def run_lambda_heatmap():
    W, H, N = 256, 256, 256
    rng = np.random.default_rng(42)
    coords: set[tuple[int, int]] = set()
    while len(coords) < N:
        coords.add((int(rng.integers(0, W)), int(rng.integers(0, H))))
    coords.update([(0, 0), (W-1, 0), (W-1, H-1), (0, H-1)])
    seeds = sorted(coords, key=lambda s: (s[0], s[1]))

    vgrid = RegularDelaunay().compute(W, H, seeds)
    tri_map, tgrid = GridTriangulation().compute(vgrid, seeds)
    n_tri = len(tri_map)

    sx = np.array([s[0] for s in seeds], dtype=np.float64)
    sy = np.array([s[1] for s in seeds], dtype=np.float64)

    tri_xa = np.zeros(n_tri); tri_ya = np.zeros(n_tri)
    tri_xb = np.zeros(n_tri); tri_yb = np.zeros(n_tri)
    tri_xc = np.zeros(n_tri); tri_yc = np.zeros(n_tri)
    for tid, (_x, _y, a, b, c) in tri_map.items():
        tri_xa[tid]=sx[a]; tri_ya[tid]=sy[a]
        tri_xb[tid]=sx[b]; tri_yb[tid]=sy[b]
        tri_xc[tid]=sx[c]; tri_yc[tid]=sy[c]

    tri_ids = tgrid[:, :, 2].astype(np.int64)
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float64)
    xa=tri_xa[tri_ids]; ya=tri_ya[tri_ids]
    xb=tri_xb[tri_ids]; yb=tri_yb[tri_ids]
    xc=tri_xc[tri_ids]; yc=tri_yc[tri_ids]
    denom = (yb-yc)*(xa-xc) + (xc-xb)*(ya-yc)
    safe_denom = np.where(np.abs(denom) < 1e-10, 1.0, denom)
    lam_a = ((yb-yc)*(xs-xc) + (xc-xb)*(ys-yc)) / safe_denom
    lam_b = ((yc-ya)*(xs-xc) + (xa-xc)*(ys-yc)) / safe_denom
    lam_c = 1.0 - lam_a - lam_b
    min_lam = np.minimum(np.minimum(lam_a, lam_b), lam_c)  # negative = outside assigned triangle

    # Build edge segments for overlay
    edge_set: set[tuple[int, int]] = set()
    segments = []
    for _, (_, _, a, b, c) in tri_map.items():
        for u, v in ((a, b), (b, c), (a, c)):
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                edge_set.add(key)
                segments.append([(sx[u], sy[u]), (sx[v], sy[v])])

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=130)
    fig.patch.set_facecolor('#111')
    for ax in axes:
        ax.set_facecolor('#111')

    # Left: min_lam heatmap — red = pixel is outside its assigned triangle
    ax = axes[0]
    vmax = 0.0;  vmin = -5.0
    im = ax.imshow(np.clip(min_lam, vmin, vmax), cmap='RdYlGn',
                   vmin=vmin, vmax=vmax, origin='upper', aspect='equal')
    ax.add_collection(LineCollection(segments, colors='white', linewidths=0.4, alpha=0.6))
    ax.scatter(sx, sy, s=4, c='white', zorder=5)
    fig.colorbar(im, ax=ax, fraction=0.03, label='min λ  (0=edge, <0=outside)')
    ax.set_title('min λ per pixel  (green=inside, red=outside assigned triangle)',
                 color='white', fontsize=10)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    # Right: violation mask — which pixels have |min_lam| > 0.5 (clearly not rasterization noise)
    ax = axes[1]
    severe = min_lam < -0.5
    ax.imshow(severe.astype(np.uint8) * 255, cmap='hot',
              vmin=0, vmax=255, origin='upper', aspect='equal')
    ax.add_collection(LineCollection(segments, colors='cyan', linewidths=0.4, alpha=0.7))
    ax.scatter(sx, sy, s=4, c='white', zorder=5)
    pct = severe.mean() * 100
    ax.set_title(f'Pixels with min λ < −0.5  ({pct:.1f}% of image)\n'
                 f'Pure rasterization aliasing would be ~0%',
                 color='white', fontsize=10)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')

    plt.tight_layout()
    plt.savefig('lambda_diagnostic.png', dpi=130, bbox_inches='tight')
    plt.show()
    print(f"Saved lambda_diagnostic.png")
    print(f"severe (min_lam < -0.5): {severe.sum()} pixels = {pct:.1f}%")
    print(f"min overall: {min_lam.min():.2f}")

def run_random_delaunay():
    W, H, N = 10, 10, 10

    # Generate N unique seed positions
    #rng = np.random.default_rng(42)
    coords: set[tuple[int, int]] = set()
    coords.update([(0, 0), (W-1, 0), (3, 3), (H-1, 3), (5, 5), (W-1, H-1), (0, H-1)])
    #while len(coords) < N:
        #coords.add((int(rng.integers(0, W)), int(rng.integers(0, H))))
    seeds = list(coords)

    # Compute Voronoi diagram
    vgrid = RegularDelaunay().compute(W, H, seeds)   # int32 (H, W, 2)


    # Compute Delaunay triangulation
    tri_map, tgrid = GridTriangulation().compute(vgrid, seeds)  # int32 (H, W, 3)
    for x in range(W):
        for y in range(H):
            print(f"Pixel ({x}, {y}): Voronoi seed_id={tgrid[y, x, 0]}, distance={tgrid[y, x, 1]}, triangle_id={tgrid[y, x, 2]}")

    # Sorted seed positions: index == seed_id assigned by the library
    sorted_seeds = sorted(seeds, key=lambda s: (s[0], s[1]))
    sx = np.array([s[0] for s in sorted_seeds])
    sy = np.array([s[1] for s in sorted_seeds])

    print(f"Seeds: {N}   Triangles: {len(tri_map)}")

    # %% — visualise
    # Perceptually-spread colormap for N discrete regions
    _hues = np.linspace(0, 1, N, endpoint=False)
    _rng2 = np.random.default_rng(7)          # shuffle so neighbours differ visually
    _hues = _hues[_rng2.permutation(N)]
    region_colors = plt.cm.hsv(_hues)
    region_cmap = mcolors.ListedColormap(region_colors)

    # Separate (shuffled) colormap for the N triangles.
    # Index 0 = red (reserved for triangle_id == -1 / outside hull).
    # Valid triangle IDs 0..n-1 are shifted to indices 1..n.
    # Hues are offset by 0.15 to keep all valid-triangle colours away from red (hue 0).
    n_tri = len(tri_map)
    _hues_t = (np.linspace(0, 1, n_tri, endpoint=False) + 0.15) % 1.0
    _hues_t = _hues_t[_rng2.permutation(n_tri)]
    tri_colors = np.vstack([[1.0, 0.0, 0.0, 1.0], plt.cm.hsv(_hues_t)])  # red first
    tri_cmap = mcolors.ListedColormap(tri_colors)
    # BoundaryNorm maps each integer 0..n_tri to exactly one colour slot.
    tri_norm = mcolors.BoundaryNorm(np.arange(-0.5, n_tri + 1.5), tri_cmap.N)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7), dpi=120)
    fig.patch.set_facecolor('#1a1a2e')
    for ax in axes:
        ax.set_facecolor('#1a1a2e')

    # ── left: Voronoi diagram ──────────────────────────────────────────────────
    ax = axes[0]
    ax.imshow(
        vgrid[:, :, 0],
        cmap=region_cmap, vmin=0, vmax=N - 1,
        origin='upper', interpolation='nearest', aspect='equal',
        alpha=0.85,
    )
    ax.scatter(sx, sy, s=6, c='white', linewidths=0, zorder=5)
    ax.set_title('Voronoi diagram', color='white', fontsize=13, pad=8)
    ax.axis('off')

    # ── right: Delaunay triangulation ─────────────────────────────────────────
    ax = axes[1]

    # Background: colour each pixel by its triangle_id.
    # Shift by +1 so -1 (outside hull) → 0 (red), valid IDs 0..n-1 → 1..n.
    tri_display = tgrid[:, :, 2].astype(np.int64) + 1
    ax.imshow(
        tri_display,
        cmap=tri_cmap, norm=tri_norm,
        origin='upper', interpolation='nearest', aspect='equal',
        alpha=0.55,
    )

    # Deduplicated triangle edges as a LineCollection
    edge_set: set[tuple[int, int]] = set()
    segments: list[list[tuple[float, float]]] = []
    for _, (_, _, a, b, c) in tri_map.items():
        for u, v in ((a, b), (b, c), (a, c)):
            key = (min(u, v), max(u, v))
            if key not in edge_set:
                edge_set.add(key)
                segments.append([
                    (sorted_seeds[u][0], sorted_seeds[u][1]),
                    (sorted_seeds[v][0], sorted_seeds[v][1]),
                ])

    ax.add_collection(LineCollection(segments, colors='white', linewidths=0.5, alpha=0.7))
    ax.scatter(sx, sy, s=8, c='white', linewidths=0, zorder=5)
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)   # keep y-down convention to match imshow
    ax.set_aspect('equal')
    ax.set_title(f'Delaunay triangulation  ({len(tri_map)} triangles)',
                color='white', fontsize=13, pad=8)
    ax.axis('off')

    plt.tight_layout(pad=1.5)
    plt.show()

if __name__ == "__main__":
    #run_lambda_heatmap()
    run_random_delaunay()