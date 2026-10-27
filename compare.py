"""Plain-distance routing vs learned-travel-time routing, across times of day."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from graph import generate_city
from algorithms import astar, dijkstra
from travel_model import train_model, predict_table, true_time, V_MAX, SPEED

HOURS = [3, 8, 13, 18, 22]


def route_stats(g, path, table, hour):
    km = pred = truth = 0.0
    for u, v in zip(path, path[1:]):
        e = next(x for x in g.adj[u] if x.v == v)
        km += e.length
        pred += table[e.idx]
        truth += true_time(e, hour)
    return km, pred, truth


N = 16
PAIRS = {"cross_city": (3 * N + 3, 12 * N + 12),    # crosses the congested centre
         "corner_to_corner": (1 * N + 1, 14 * N + 14)}


def run(g, model, src, dst, tag):
    print(f"\n===== OD pair: {tag}  (node {src} -> node {dst}) =====")

    # 1) plain distance, Dijkstra vs A*
    dist_cost = lambda e: e.length
    d = dijkstra(g, src, dst, dist_cost)
    a = astar(g, src, dst, dist_cost, lambda x, goal: g.euclid(x, goal))
    print(f"[Distance cost] Dijkstra: {d.cost:.2f} km, expanded {d.expanded} nodes | "
          f"A*: {a.cost:.2f} km, expanded {a.expanded} nodes\n")
    dist_path = a.path

    # 2) learned travel time at different hours
    print(f"{'Hour':>4} | {'Dist-route':>22} | {'Time-route (learned)':>22} | {'Saved':>6} | Changed | A* vs Dij expanded")
    print(f"{'':>4} | {'km':>8} {'true min':>12} | {'km':>8} {'true min':>12} |        |         |")
    print("-" * 95)
    plots = {}
    for hr in HOURS:
        table = predict_table(model, g, hr)
        cost = lambda e: table[e.idx]
        h = lambda x, goal: g.euclid(x, goal) / V_MAX * 60 * 0.85   # optimistic = admissible
        dj = dijkstra(g, src, dst, cost)
        As = astar(g, src, dst, cost, h)
        assert abs(dj.cost - As.cost) < 1e-6
        km_d, _, truth_d = route_stats(g, dist_path, table, hr)
        km_t, _, truth_t = route_stats(g, As.path, table, hr)
        saved = 100 * (truth_d - truth_t) / truth_d
        print(f"{hr:>4} | {km_d:>8.2f} {truth_d:>12.1f} | {km_t:>8.2f} {truth_t:>12.1f} | "
              f"{saved:>5.1f}% | {'YES' if As.path != dist_path else 'no':>7} | {As.expanded} vs {dj.expanded}")
        plots[hr] = (table, As.path)

    # 3) figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, hr in zip(axes, [3, 8, 18]):
        table, tpath = plots[hr]
        for e in g.edges:
            if e.u < e.v:
                (x1, y1), (x2, y2) = g.pos[e.u], g.pos[e.v]
                free = e.length / SPEED[e.road] * 60
                slow = min(table[e.idx] / free, 4) / 4          # 0 free-flow .. 1 gridlock
                ax.plot([x1, x2], [y1, y2], color=plt.cm.RdYlGn_r(slow), lw=1.0 + (e.road == "highway") * 2)
        for path, c, ls, lab in ((dist_path, "blue", "--", "Shortest distance"),
                                 (tpath, "black", "-", "Learned fastest")):
            ax.plot([g.pos[x][0] for x in path], [g.pos[x][1] for x in path], color=c, ls=ls, lw=2.6, label=lab)
        ax.scatter(*g.pos[src], c="blue", s=90, zorder=5); ax.scatter(*g.pos[dst], c="purple", s=90, zorder=5)
        ax.set_title(f"{hr:02d}:00  (road colour = predicted slowdown)")
        ax.set_aspect("equal"); ax.axis("off"); ax.legend(loc="lower right")
    plt.tight_layout(pad=2.0)
    plt.savefig(f"route_comparison_{tag}.png", dpi=130)
    plt.close(fig)
    print(f"Saved route_comparison_{tag}.png")


def main():
    g = generate_city()
    print(f"Network: {len(g.pos)} intersections, {len(g.edges)//2} road segments")
    model, m = train_model(g)
    print(f"Travel-time model (RandomForest): MAE={m['MAE_min']:.3f} min/segment, R2={m['R2']:.3f}")
    for tag, (s, t) in PAIRS.items():
        run(g, model, s, t, tag)


if __name__ == "__main__":
    main()
