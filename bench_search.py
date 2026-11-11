"""Scale benchmark for ResonanceIndex3D on synthetic nodes.
Usage:  python bench_search.py 1000 10000 100000
Per size (fresh process): build time, RAM growth, p50/p95 latency of the engine vs a naive linear
name scan, recall@10 against the exact (full-postings) result, and precision@10 against ground truth
(does the returned node's name contain the queried word?)."""
import json, random, statistics as st, subprocess, sys, time
import networkx as nx, numpy as np
import app as A

W = ("kalman filter lidar slam yolo anchor transformer cuda kernel latency cache simd lock free queue ring buffer "
 "graph dijkstra segment tree fenwick convex hull plc ladder modbus profibus servo torque jacobian dynamics "
 "quaternion euler spline voxel octree mesh cesium tiles geospatial wgs84 ellipsoid raster clinical ner icd "
 "token embedding quantization gguf llama attention rope lora diffusion tof depth point cloud calibration "
 "homography stereo optical flow tracker mpc pid observer lyapunov monte carlo pde fem lattice boltzmann "
 "fft wavelet sparse eigen svd cholesky gauss newton adam sgd batchnorm resnet mobilenet nms iou anchorfree "
 "ros dds zeromq mqtt grpc protobuf arena allocator numa affinity prefetch branchless inline template").split()

def make(n):
    r = random.Random(0)
    ids = ["_".join(r.choice(W).capitalize() for _ in range(r.randint(2, 3))) + "_%d" % i for i in range(n)]
    g = nx.Graph(); g.add_nodes_from(ids)
    for i in range(n):
        for _ in range(2): g.add_edge(ids[i], ids[r.randrange(n)])
    return ids, {i: r.choice(["Core", "Robotics", "AI", "Math", "Systems"]) for i in ids}, {i: "/none/" + i for i in ids}, g

def run(n, nq=300):
    ids, subj, fold, g = make(n)
    r0 = A._rss_mb(); t = time.perf_counter()
    e = A.ResonanceIndex3D(ids, subj, fold, g)
    build = time.perf_counter() - t; r1 = A._rss_mb()
    low = [x.lower() for x in ids]; rnd = random.Random(1)
    lat, scan, cand, rec, prec = [], [], [], [], []
    for _ in range(nq):
        q = rnd.choice(W)
        res, s = e.search(q); lat.append(s["query_ms"]); cand.append(s["candidates"])
        t = time.perf_counter(); _ = [i for i, x in enumerate(low) if q in x]; scan.append((time.perf_counter() - t) * 1000)
        ex, _s = e.search(q, exact=True)
        if ex:
            top = {a for a, _ in ex[:10]}; rec.append(len(top & {a for a, _ in res[:10]}) / len(top))
        top = [a for a, _ in res[:10]]
        if top: prec.append(sum(q in a.lower() for a in top) / len(top))
    p = lambda a, k: round(float(np.percentile(a, k)), 3)
    return dict(nodes=n, build_s=round(build, 1), ram_per_node_KB=round((r1 - r0) * 1024 / n, 2), index_MB=round(e.bucket_stats()["index_bytes"] / 1048576, 1),
                p50_ms=p(lat, 50), p95_ms=p(lat, 95), cand_mean=int(st.mean(cand)), linear_scan_p50_ms=p(scan, 50),
                recall10_vs_exact=round(st.mean(rec), 3), precision10_vs_truth=round(st.mean(prec), 3))

if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:
        print(json.dumps(run(int(sys.argv[2])))); sys.exit()
    rows = [json.loads(subprocess.run([sys.executable, __file__, "--one", n], capture_output=True, text=True).stdout.strip().splitlines()[-1])
            for n in (sys.argv[1:] or ["1000", "10000", "100000"])]
    print("  ".join("%-20s" % k for k in rows[0]))
    for r in rows: print("  ".join("%-20s" % r[k] for k in r))
