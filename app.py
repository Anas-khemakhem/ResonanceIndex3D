#!/usr/bin/env python3
"""
Jarvis 3D Brain  -  edge-native neural memory & knowledge graph.

Run:   pip install -r requirements.txt
       python app.py            ->  http://127.0.0.1:5000

Design notes
  * RAM:  only node metadata and compact numpy posting arrays live in memory (~0.4 KB/node).
          Document text is NEVER preloaded; SSD folders are read lazily for matched/clicked nodes.
  * Search: Resonance index = phase-free "quantum-inspired" amplitudes on a sparse feature space.
          Each node has a wave Psi_i (name/subject/file-name atoms, TF-IDF weighted, plus 0.35 x the
          token atoms of its neighbours, precomputed at startup = neighbourhood superposition).
          The query is a wave q; the Born score is P = |<q|Psi_i>|^2.  Postings are stored in
          impact order (highest weight first) and at most MAX_POSTINGS are read per query atom, so
          query cost is bounded and independent of the number of nodes.  No BFS/DFS at query time.
"""
import ctypes
import hashlib
import math
import os
import re
import subprocess
import sys
import time

import networkx as nx
import numpy as np
from flask import Flask, abort, jsonify, request, send_from_directory

# =====================================================================================
#  PART 1 - STATIC GRAPH CONFIGURATION  (edit here, restart the server)
# =====================================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SSD_ROOT = os.path.join(BASE_DIR, "jarvis_ssd_memory")


def _ssd(name):
    """Dedicated SSD folder for a node:  ./jarvis_ssd_memory/<Node_Name>/"""
    return os.path.join(SSD_ROOT, name)


SUBJECT_COLORS = {
    "Core": "#ffd166",
    "Robotics & Hardware": "#ff8a5b",
    "AI & Vision": "#4cc9f0",
    "Math & Algorithms": "#b794f6",
    "Systems": "#5eead4",
}

MEMORY_NODES = [
    # --- Core -------------------------------------------------------------------------
    {"id": "Jarvis",                 "subject": "Core",                "ssd": _ssd("Jarvis")},
    # --- Robotics & Hardware ----------------------------------------------------------
    {"id": "Robotics",               "subject": "Robotics & Hardware", "ssd": _ssd("Robotics")},
    {"id": "Inverse_Kinematics",     "subject": "Robotics & Hardware", "ssd": _ssd("Inverse_Kinematics")},
    {"id": "ToF_3D_Mapping",         "subject": "Robotics & Hardware", "ssd": _ssd("ToF_3D_Mapping")},
    {"id": "Robotic_Arm",            "subject": "Robotics & Hardware", "ssd": _ssd("Robotic_Arm")},
    # --- AI & Vision ------------------------------------------------------------------
    {"id": "Anchor_Free_YOLO",       "subject": "AI & Vision",         "ssd": _ssd("Anchor_Free_YOLO")},
    {"id": "Local_LLMs",             "subject": "AI & Vision",         "ssd": _ssd("Local_LLMs")},
    {"id": "Clinical_NLP",           "subject": "AI & Vision",         "ssd": _ssd("Clinical_NLP")},
    # --- Math & Algorithms ------------------------------------------------------------
    {"id": "Competitive_Programming", "subject": "Math & Algorithms",  "ssd": _ssd("Competitive_Programming")},
    {"id": "Graph_Theory",           "subject": "Math & Algorithms",   "ssd": _ssd("Graph_Theory")},
    {"id": "Computational_Physics",  "subject": "Math & Algorithms",   "ssd": _ssd("Computational_Physics")},
    # --- Systems ----------------------------------------------------------------------
    {"id": "Low_Latency_Cpp",        "subject": "Systems",             "ssd": _ssd("Low_Latency_Cpp")},
    {"id": "Cesium_Geospatial",      "subject": "Systems",             "ssd": _ssd("Cesium_Geospatial")},
    {"id": "Industrial_PLC",         "subject": "Systems",             "ssd": _ssd("Industrial_PLC")},
]

MEMORY_LINKS = [
    # Jarvis hub -> one anchor per subject
    ("Jarvis", "Robotics"),
    ("Jarvis", "Local_LLMs"),
    ("Jarvis", "Competitive_Programming"),
    ("Jarvis", "Low_Latency_Cpp"),
    # Robotics & Hardware cluster
    ("Robotics", "Inverse_Kinematics"),
    ("Robotics", "ToF_3D_Mapping"),
    ("Robotics", "Robotic_Arm"),
    ("Inverse_Kinematics", "Robotic_Arm"),
    ("Robotic_Arm", "Industrial_PLC"),
    # AI & Vision cluster
    ("Anchor_Free_YOLO", "Local_LLMs"),
    ("Local_LLMs", "Clinical_NLP"),
    ("ToF_3D_Mapping", "Anchor_Free_YOLO"),
    # Math & Algorithms cluster
    ("Competitive_Programming", "Graph_Theory"),
    ("Graph_Theory", "Computational_Physics"),
    ("Computational_Physics", "Inverse_Kinematics"),
    # Systems cluster
    ("Low_Latency_Cpp", "Industrial_PLC"),
    ("Low_Latency_Cpp", "Anchor_Free_YOLO"),
    ("Low_Latency_Cpp", "Computational_Physics"),
    ("ToF_3D_Mapping", "Cesium_Geospatial"),
    ("Graph_Theory", "Cesium_Geospatial"),
]

# 3D placement on the cortex:  node_id -> (hemisphere, (lateral, up, front))
#   hemisphere  -1 = left, +1 = right
#   direction   lateral>0 = outward from the midline, up>0 = dorsal, front>0 = frontal pole
# Special entries may be a plain (x, y, z) tuple of 3 floats for fixed placement.
BRAIN_PLACEMENT = {
    "Jarvis":                  (0.0, 0.2, 0.3),                 # thalamic hub in the midline fissure
    "Robotics":                (-1, (0.35, 0.55, 0.78)),        # left frontal lobe
    "Inverse_Kinematics":      (-1, (0.85, 0.45, 0.25)),
    "ToF_3D_Mapping":          (-1, (0.50, -0.15, 0.95)),
    "Robotic_Arm":             (-1, (0.95, -0.20, 0.05)),
    "Competitive_Programming": (-1, (0.35, 0.90, -0.20)),       # left parietal lobe
    "Graph_Theory":            (-1, (0.80, 0.50, -0.55)),
    "Computational_Physics":   (-1, (0.35, 0.40, -0.90)),       # left occipital pole
    "Anchor_Free_YOLO":        (1, (0.35, 0.35, -0.90)),        # right occipital (vision)
    "Local_LLMs":              (1, (0.80, 0.50, -0.45)),
    "Clinical_NLP":            (1, (0.40, 0.90, -0.30)),
    "Low_Latency_Cpp":         (1, (0.35, 0.55, 0.78)),         # right frontal lobe
    "Cesium_Geospatial":       (1, (0.90, -0.45, 0.20)),        # right temporal lobe
    "Industrial_PLC":          (0.9, -1.55, -3.2),              # cerebellum
}

TOP_K = 30              # max matches returned per query
MAX_POSTINGS = 256      # postings read per query atom (impact-ordered) -> O(1) work per query
MAX_NEIGHBORS = 32      # neighbours blended into a node wave (caps hub cost)
NEIGHBOR_WEIGHT = 0.35  # neighborhood superposition strength
MIN_RESONANCE = 0.02    # minimum Born probability to count as a match
HOST = os.environ.get("JARVIS_HOST", "127.0.0.1")
PORT = int(os.environ.get("JARVIS_PORT", "5000"))

# =====================================================================================
#  Wave-vector encoding primitives (pure NumPy)
# =====================================================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text):
    return _TOKEN_RE.findall(text.lower())


def _features(text, scale=1.0, out=None):
    """Token atoms (1) + 4-char prefix atoms (0.8) + padded character trigram atoms (0.5)."""
    out = {} if out is None else out
    for tok in _tokens(text):
        a = "t:" + tok
        out[a] = out.get(a, 0.0) + scale
        if len(tok) >= 4:                                    # prefix atom: "robot" ~ "robotic", "robotics"
            pa = "p:" + tok[:4]
            out[pa] = out.get(pa, 0.0) + 0.8 * scale
        p = "^" + tok + "$"
        for i in range(len(p) - 2):
            g = "g:" + p[i:i + 3]
            out[g] = out.get(g, 0.0) + 0.5 * scale
    return out


def _list_names(folder):
    """File NAMES only (metadata) - contents are never read here."""
    try:
        return sorted(e.name for e in os.scandir(folder) if e.is_file())
    except OSError:
        return []


# =====================================================================================
#  PART 2a - ResonanceIndex3D  (impact-ordered sparse wave index, Born-rule scoring)
# =====================================================================================
class ResonanceIndex3D:
    """
    Psi_i  = norm( own_atoms_i + w_nb * sum_{j in N(i)} token_atoms_j )     (TF-IDF weighted, startup)
    q      = norm( query atoms )                                             (same weighting)
    score  = P_i = |<q | Psi_i>|^2                                           (Born rule)
    Postings per atom are sorted by weight descending; a query reads <= max_postings per atom.
    """

    def __init__(self, ids, subjects, folders, graph, neighbor_weight=NEIGHBOR_WEIGHT,
                 min_resonance=MIN_RESONANCE, max_postings=MAX_POSTINGS, top_k=TOP_K):
        self.ids = list(ids)
        self.index = {nid: i for i, nid in enumerate(self.ids)}
        self.min_resonance, self.max_postings, self.top_k = min_resonance, max_postings, top_k
        n = len(self.ids)

        vocab, is_tok = {}, []
        own_k, own_w = [], []
        for nid in self.ids:                                   # pass 1: own atoms (names only, no file reads)
            feats = {}
            _features(nid, 1.0, feats)
            _features(subjects[nid], 0.6, feats)
            for fname in _list_names(folders[nid]):
                _features(os.path.splitext(fname)[0], 0.5, feats)
            ks = []
            for a in feats:
                k = vocab.get(a)
                if k is None:
                    k = vocab[a] = len(vocab)
                    is_tok.append(a[0] == "t")
                ks.append(k)
            own_k.append(np.array(ks, dtype=np.int32))
            own_w.append(np.fromiter(feats.values(), dtype=np.float32, count=len(ks)))
        V = len(vocab)
        is_tok = np.array(is_tok, dtype=bool)
        df = np.bincount(np.concatenate(own_k), minlength=V) if n else np.zeros(V, dtype=np.int64)
        self.idf = np.log1p(n / np.maximum(df, 1)).astype(np.float32)
        self.idf_unknown = float(math.log1p(n))
        self.vocab = vocab

        A, NODE, W = [], [], []
        for i, nid in enumerate(self.ids):                     # pass 2: neighbourhood superposition
            pk, pw = [own_k[i]], [own_w[i]]
            for nb in list(graph.neighbors(nid))[:MAX_NEIGHBORS]:
                j = self.index[nb]
                m = is_tok[own_k[j]]
                pk.append(own_k[j][m])
                pw.append(own_w[j][m] * neighbor_weight)
            k = np.concatenate(pk)
            u, inv = np.unique(k, return_inverse=True)
            w = np.bincount(inv, weights=np.concatenate(pw), minlength=len(u)).astype(np.float32) * self.idf[u]
            w /= max(float(np.sqrt((w * w).sum())), 1e-12)
            A.append(u.astype(np.int32)); W.append(w); NODE.append(np.full(len(u), i, dtype=np.int32))
        atoms, nodes, wts = np.concatenate(A), np.concatenate(NODE), np.concatenate(W)
        del A, NODE, W, own_k, own_w
        order = np.lexsort((-wts, atoms))                      # by atom, then weight descending
        self.p_nodes = nodes[order]
        self.p_w = wts[order]
        self.ptr = np.zeros(V + 1, dtype=np.int64)
        np.cumsum(np.bincount(atoms, minlength=V), out=self.ptr[1:])

    def _query(self, query):
        feats = _features(query)
        ks, ws, unk = [], [], 0.0
        for a, tf in feats.items():
            k = self.vocab.get(a)
            if k is None:
                unk += (tf * self.idf_unknown) ** 2
            else:
                ks.append(k)
                ws.append(tf * float(self.idf[k]))
        if not ks:
            return None, None
        w = np.array(ws, dtype=np.float32)
        w /= max(math.sqrt(float((w * w).sum()) + unk), 1e-12)
        return np.array(ks, dtype=np.int64), w

    def _score(self, ks, qw, limit):
        ptr, parts_n, parts_c = self.ptr, [], []
        for k, w in zip(ks.tolist(), qw.tolist()):
            s, e = int(ptr[k]), int(ptr[k + 1])
            if limit:
                e = min(e, s + limit)
            parts_n.append(self.p_nodes[s:e])
            parts_c.append(self.p_w[s:e] * w)
        nodes = np.concatenate(parts_n)
        uniq, inv = np.unique(nodes, return_inverse=True)
        amp = np.bincount(inv, weights=np.concatenate(parts_c), minlength=len(uniq))   # <q|Psi_i>
        prob = amp * amp                                                               # Born rule
        keep = np.nonzero(prob >= self.min_resonance)[0]
        if len(keep) > self.top_k:
            keep = keep[np.argpartition(-prob[keep], self.top_k - 1)[:self.top_k]]
        keep = keep[np.argsort(-prob[keep])]
        return [(self.ids[int(uniq[j])], float(prob[j])) for j in keep], len(uniq)

    def search(self, query, exact=False):
        """Returns ([(node_id, born_probability), ...] desc, stats).  exact=True reads full postings (baseline)."""
        t0 = time.perf_counter()
        ks, qw = self._query(query)
        if ks is None:
            return [], {"candidates": 0, "probes": 0, "query_ms": (time.perf_counter() - t0) * 1000.0}
        res, cands = self._score(ks, qw, 0 if exact else self.max_postings)
        return res, {"candidates": cands, "probes": int(len(ks)), "query_ms": (time.perf_counter() - t0) * 1000.0}

    def bucket_stats(self):
        return {"atoms": len(self.vocab), "postings": int(len(self.p_nodes)),
                "index_bytes": int(self.p_nodes.nbytes + self.p_w.nbytes + self.ptr.nbytes + self.idf.nbytes)}


# =====================================================================================
#  PART 2b - 3D brain topology (deterministic)
# =====================================================================================
HEMI_X = 1.0
AX_A, AX_UP, AX_DOWN, AX_C = 1.85, 2.15, 1.35, 3.3     # hemisphere ellipsoid semi-axes
_GOLDEN = math.pi * (3.0 - math.sqrt(5.0))


def _unit(lat, up, fr):
    n = math.sqrt(lat * lat + up * up + fr * fr) or 1.0
    return lat / n, up / n, fr / n


def _cortex_radius(d):
    lat, up, fr = d
    b = AX_UP if up >= 0 else AX_DOWN
    r = 1.0 / math.sqrt((lat / AX_A) ** 2 + (up / b) ** 2 + (fr / AX_C) ** 2)
    bulge = 1.0 + 0.10 * math.exp(-((lat - 0.8) ** 2 + (up + 0.45) ** 2 + (fr - 0.2) ** 2) / 0.12)  # temporal lobe
    return r * bulge


def _cortex_point(hemi, d, depth=1.0, gyri=False):
    lat, up, fr = d
    r = _cortex_radius(d) * depth
    if gyri:
        r *= 1.0 + 0.032 * (math.sin(7.0 * lat + 2.0 * fr) * math.cos(6.0 * up - 1.5 * fr)
                            + 0.6 * math.sin(11.0 * fr + 3.0 * lat))
    x = max(HEMI_X + lat * r, 0.14)
    return hemi * x, up * r, fr * r


def compute_layout():
    pos = {}
    for node in MEMORY_NODES:
        nid = node["id"]
        spec = BRAIN_PLACEMENT.get(nid)
        if spec is None:                                   # deterministic fallback for new nodes
            h = int.from_bytes(hashlib.blake2b(nid.encode(), digest_size=8).digest(), "little")
            hemi = 1 if h & 1 else -1
            a = ((h >> 1) % 1000) / 1000.0 * 2.0 * math.pi
            y = (((h >> 11) % 1000) / 1000.0) * 1.6 - 0.6
            spec = (hemi, (0.5 + 0.4 * math.cos(a), y, math.sin(a)))
        if len(spec) == 3:
            pos[nid] = tuple(float(v) for v in spec)
        else:
            hemi, d = spec
            x, y, z = _cortex_point(hemi, _unit(*d), depth=0.84)
            pos[nid] = (x, y, z)
    return pos


def brain_shell():
    """Decorative cortex / cerebellum / brainstem point cloud (flat list of floats)."""
    pts = []
    n = 1500
    for hemi in (-1, 1):
        for k in range(n):
            y = 1.0 - 2.0 * (k + 0.5) / n
            rad = math.sqrt(max(0.0, 1.0 - y * y))
            th = _GOLDEN * k
            d = (math.cos(th) * rad, y, math.sin(th) * rad)      # (lateral, up, front)
            if d[0] < -0.30:                                     # open longitudinal fissure
                continue
            pts.extend(_cortex_point(hemi, d, 1.0, gyri=True))
    cn = 380
    for k in range(cn):
        y = 1.0 - 2.0 * (k + 0.5) / cn
        if y > 0.75:
            continue
        rad = math.sqrt(max(0.0, 1.0 - y * y))
        th = _GOLDEN * k
        s = 1.0 + 0.05 * math.sin(26.0 * y)                      # folia ridges
        pts.extend((1.3 * math.cos(th) * rad * s, -1.55 + 0.75 * y * s, -2.75 + 0.9 * math.sin(th) * rad * s))
    for k in range(120):
        s = k / 119.0
        a = _GOLDEN * k
        rr = 0.38 * (1.0 - 0.3 * s)
        pts.extend((rr * math.cos(a), -0.9 - 2.2 * s, -0.5 - 0.6 * s + rr * math.sin(a)))
    return [round(v, 3) for v in pts]


# =====================================================================================
#  Seed notes - written ONLY when a node's SSD folder does not exist yet
# =====================================================================================
def _seed_notes():
    return {
        "Jarvis": {
            "jarvis_system_architecture_overview.md": """# Jarvis system architecture

Jarvis is the central orchestrator. Every capability is a small module that talks over a local message bus
(ZeroMQ PUB/SUB on localhost, JSON frames, topic = module name).

- Perception modules publish sensor frames: camera, ToF depth, microphone.
- Planner subscribes to goals and publishes motion and speech intents.
- Memory service (this 3D brain) answers keyword queries in O(1) via phase-encoded hashing.
- Safety supervisor can veto any actuator command within one control tick (5 ms budget).

Design rule: no module blocks the bus; heavy work runs in a worker process and replies asynchronously.
""",
            "jarvis_voice_wake_word_pipeline.txt": """Voice wake-word pipeline

Audio: 16 kHz mono, 30 ms frames with 10 ms hop.
Features: 40-bin log-mel spectrogram, per-utterance mean/variance normalisation.
Wake word: small depthwise-separable CNN, INT8, ~120 KB, runs on one CPU core below 3 percent load.
After a wake hit the pipeline opens a 6 second window for speech-to-text, then routes text to Local_LLMs.
False-accept target: under 1 per 10 hours of TV background noise.
""",
        },
        "Robotics": {
            "robotics_control_loop_pid_tuning.md": """# Control loop and PID tuning

Joint controllers run at 1 kHz with a fixed-step timer; the planner runs at 50 Hz.

PID tips
- Filter the derivative term (first order low-pass, N = 10 to 20) to avoid noise kicks.
- Add anti-windup by clamping the integrator or using back-calculation when the output saturates.
- Ziegler-Nichols gives a starting point: Kp = 0.6 Ku, Ti = Tu / 2, Td = Tu / 8. Then hand-tune for less overshoot.
- Feed-forward gravity compensation removes most of the steady error on vertical joints.
""",
            "robotics_ros2_node_topology_notes.txt": """ROS 2 node topology notes

/camera_driver -> /perception -> /planner -> /arm_controller -> /hardware_interface
DDS QoS: sensor streams use BEST_EFFORT + KEEP_LAST(1); commands use RELIABLE + KEEP_LAST(10).
Use composable nodes with intra-process communication to avoid serialization of large images.
Time sync: stamp frames at capture, use message_filters ApproximateTime for RGB + depth fusion.
Launch files keep parameters in YAML so the same stack runs in simulation and on the real arm.
""",
        },
        "Inverse_Kinematics": {
            "inverse_kinematics_jacobian_damped_least_squares.md": """# Jacobian inverse kinematics with damped least squares

Task error e = x_target - f(q). The Jacobian J(q) maps joint velocities to end-effector twist.

Damped least squares update:
    dq = J^T (J J^T + lambda^2 I)^-1 e
Small lambda tracks accurately; large lambda is stable near singularities. Adapt lambda from the smallest
singular value: lambda^2 = lambda_max^2 * (1 - sigma_min / eps)^2 when sigma_min < eps.

Iterate until |e| < 1e-4 m or 50 iterations. Warm-start from the previous solution for smooth trajectories.
""",
            "inverse_kinematics_singularity_avoidance_notes.txt": """Singularity avoidance notes

Singular configurations: wrist alignment (joint 5 near 0), elbow fully extended, shoulder over the base axis.
Manipulability measure w = sqrt(det(J J^T)); monitor it and slow the Cartesian speed when w drops.
Null-space projection lets a 7-DOF arm optimise secondary goals: dq = J+ e + (I - J+ J) dq0,
where dq0 = gradient of joint-limit distance. For 6-DOF arms, re-plan through a via-point instead.
""",
        },
        "ToF_3D_Mapping": {
            "tof_depth_point_cloud_calibration.md": """# ToF depth to point cloud and calibration

Continuous-wave time-of-flight: depth d = c * phase / (4 * pi * f_mod). Unambiguous range = c / (2 * f_mod),
so a 20 MHz modulation gives 7.5 m.

Back-projection with pinhole intrinsics: X = (u - cx) * d / fx, Y = (v - cy) * d / fy, Z = d.
Calibration steps: intrinsics with a checkerboard, per-pixel depth offset (wiggling error) with a flat wall at
several distances, extrinsics to the robot base through hand-eye calibration.
Voxel-grid downsample at 1 cm before registration with ICP.
""",
            "tof_multipath_interference_filtering.txt": """Multipath interference filtering

Corners and glossy floors create multipath: the sensor sees a mix of direct and bounced light and over-estimates depth.
Mitigation:
- Use two modulation frequencies and reject pixels where the unwrapped depths disagree by more than 3 cm.
- Flying-pixel filter: drop points whose normal is nearly perpendicular to the viewing ray.
- Temporal median over 5 frames for static scenes.
Amplitude below a threshold means low confidence; mask those pixels before meshing.
""",
        },
        "Robotic_Arm": {
            "robotic_arm_6dof_dh_parameters.md": """# 6-DOF arm Denavit-Hartenberg parameters

Each link i uses (theta_i, d_i, a_i, alpha_i). Transform:
    T_i = Rz(theta_i) * Tz(d_i) * Tx(a_i) * Rx(alpha_i)
Forward kinematics is the product T_1 ... T_6.

Example (hobby arm): link lengths 0.12 m, 0.25 m, 0.22 m, wrist spherical so IK has a closed form.
Joint limits are stored in radians in config and enforced in the controller, not only in the planner.
Verify the model by commanding five random poses and comparing predicted vs measured tool position (target under 2 mm).
""",
            "robotic_arm_servo_torque_limits.txt": """Servo torque limits

Shoulder servo stall torque 35 kg*cm at 12 V. Keep continuous load under 60 percent of stall.
Payload check: tau = m * g * L at full reach; for 0.5 kg at 0.45 m that is about 2.2 N*m (22 kg*cm) - acceptable.
Add a gas spring or counterweight on the shoulder when payload grows.
Current limit in firmware trips at 2.5 A for 200 ms to protect the gearbox. Thermal derating after 10 minutes continuous work.
""",
        },
        "Anchor_Free_YOLO": {
            "anchor_free_yolo_center_sampling_head.md": """# Anchor-free YOLO: center sampling and decoupled head

Anchor-free detectors predict the box as distances (l, t, r, b) from each grid cell to the box sides instead of
offsets to predefined anchors. This removes anchor hyper-parameters and clustering.

- Decoupled head: separate branches for classification and regression converge faster.
- Center sampling: only cells near the object center are positives (radius about 2.5 strides).
- Label assignment: SimOTA (YOLOX) or Task-Aligned Assigner (YOLOv8) picks positives by cost = cls + IoU.
- Loss: BCE for classification, CIoU or DFL + GIoU for boxes.
NMS at 0.5 IoU, confidence 0.25 by default.
""",
            "anchor_free_yolo_int8_quantization_cpu_inference.txt": """INT8 quantization and CPU inference

Target: integrated-graphics laptop, no discrete GPU. Export to ONNX, run with OpenVINO or ONNX Runtime (CPU, 4 threads).
Post-training static quantization with 300 calibration images typically costs under 1 mAP and gives 2 to 3x speed-up.
Use the nano model at 416 x 416: about 20 to 30 ms per frame on a 12th-gen i7 U core set.
Fuse Conv + BN + SiLU, keep the detection head output in FP16/FP32 if boxes jitter.
""",
        },
        "Local_LLMs": {
            "local_llms_gguf_quantization_cpu_offload.md": """# Local LLMs: GGUF quantization on CPU

llama.cpp loads GGUF files with memory mapping, so weights page in from the SSD on demand.
Quant levels: Q8_0 (near lossless), Q5_K_M, Q4_K_M (best size/quality trade-off), Q3_K_M (noticeable drop).
Rule of thumb: bytes = parameters * bits / 8. A 7B model at Q4_K_M is about 4.4 GB.
Speed on CPU is memory-bandwidth bound: tokens/s ~ bandwidth / model size. Use 6 to 8 threads (performance cores).
With integrated graphics only, keep n_gpu_layers = 0 and enable AVX2 builds.
""",
            "local_llms_rag_prompt_context_window.txt": """RAG prompt and context window notes

Retrieval: ask the 3D brain for matching nodes, read only the top 3 files, chunk to 400 tokens with 50 overlap.
Prompt layout: system rules, retrieved context with file names as citations, then the user question.
Keep total context under 4096 tokens to hold KV cache near 512 MB for a 7B model.
Instruct the model to answer only from context and to say 'not found' otherwise. Log the cited file names for audit.
""",
        },
        "Clinical_NLP": {
            "clinical_nlp_icd10_entity_extraction.md": """# Clinical NLP: entity extraction and ICD-10 coding

Pipeline: sentence split -> tokenization -> NER (problems, medications, tests, procedures) -> negation/context -> concept linking.
Linking targets: ICD-10-CM for diagnoses, SNOMED CT for findings, RxNorm for drugs.
Negation and uncertainty: NegEx / ConText rules ('denies chest pain', 'no evidence of pneumonia', 'rule out sepsis').
Abbreviation disambiguation matters: 'MS' can be multiple sclerosis, mitral stenosis or morphine sulfate.
Evaluate with entity-level F1 and a clinician-reviewed sample of 200 notes.
""",
            "clinical_nlp_deidentification_phi_rules.txt": """De-identification and PHI rules

Under the HIPAA Safe Harbor method, 18 identifier types must be removed: names, geographic units smaller than a state,
all dates except year, phone, fax, email, SSN, medical record numbers, account numbers, device identifiers, URLs, IPs, biometrics,
full-face photos and any other unique code.
Approach: regex for structured identifiers + a token-classification model for names and places, then replace with typed placeholders.
Always process notes locally; never send raw clinical text to external services.
""",
        },
        "Competitive_Programming": {
            "competitive_programming_segment_tree_templates.md": """# Segment tree templates

Point update, range query in O(log n); build in O(n).
Iterative bottom-up version (size n, array t of length 2n):
    update: p += n; t[p] = v; while p > 1: p >>= 1; t[p] = t[2p] + t[2p+1]
    query [l, r): res = 0; for l += n, r += n; l < r; l >>= 1, r >>= 1: if l&1: res += t[l++]; if r&1: res += t[--r]
Lazy propagation handles range add / range sum: keep a pending tag per node, push down before recursing.
Related: Fenwick tree for prefix sums, sparse table for static range minimum (O(1) query).
""",
            "competitive_programming_dp_bitmask_patterns.txt": """DP and bitmask patterns

Bitmask DP over subsets: dp[mask][last]. Travelling salesman in O(n^2 * 2^n), feasible up to n = 20.
Subset enumeration trick: for (sub = mask; sub; sub = (sub - 1) & mask) gives all submasks; total over all masks is 3^n.
Knapsack in O(nW) with a rolling 1D array iterating weights downward.
Digit DP: state (position, tight, started, remainder). Always memoise on the non-tight states only.
Checklist: define state, transition, base case, order of evaluation, answer cell.
""",
        },
        "Graph_Theory": {
            "graph_theory_dijkstra_vs_bellman_ford.md": """# Dijkstra vs Bellman-Ford

Dijkstra: non-negative weights, O((V + E) log V) with a binary heap. Greedy: once popped, a distance is final.
Bellman-Ford: handles negative edges, O(V * E); a V-th relaxation round that still improves reveals a negative cycle.
SPFA is a queue-based Bellman-Ford with good average behaviour but a bad worst case.
All-pairs: Floyd-Warshall O(V^3); Johnson reweights with Bellman-Ford then runs Dijkstra from every vertex.
Unweighted graphs: plain BFS gives shortest paths in O(V + E). A* adds an admissible heuristic.
""",
            "graph_theory_max_flow_dinic_min_cut.txt": """Max flow with Dinic and the min-cut theorem

Dinic: build a level graph by BFS, then send blocking flows by DFS with a current-arc pointer. Complexity O(V^2 * E);
on unit-capacity bipartite graphs it is O(E * sqrt(V)).
Max-flow min-cut: the maximum flow value equals the capacity of the minimum s-t cut. After the algorithm ends,
vertices reachable from s in the residual graph form the source side of the cut.
Applications: bipartite matching, project selection, image segmentation, edge-disjoint paths.
""",
        },
        "Computational_Physics": {
            "computational_physics_runge_kutta_integrators.md": """# Runge-Kutta integrators

Classical RK4 for y' = f(t, y):
    k1 = f(t, y); k2 = f(t + h/2, y + h k1/2); k3 = f(t + h/2, y + h k2/2); k4 = f(t + h, y + h k3)
    y_next = y + h (k1 + 2 k2 + 2 k3 + k4) / 6
Local error O(h^5), global error O(h^4). Use adaptive Dormand-Prince (RK45) when the dynamics are stiff-free but uneven.
For long Hamiltonian simulations prefer symplectic integrators (velocity Verlet) since they conserve energy over time.
Typical use here: arm dynamics, orbital mechanics and spring-mass models.
""",
            "computational_physics_finite_difference_heat_equation.txt": """Finite difference for the 1D heat equation

u_t = alpha * u_xx. Explicit FTCS scheme: u[i]^(n+1) = u[i]^n + r (u[i+1]^n - 2 u[i]^n + u[i-1]^n), r = alpha dt / dx^2.
Stability requires r <= 0.5. Crank-Nicolson is unconditionally stable and second order in time but needs a tridiagonal solve (Thomas algorithm, O(N)).
Boundary conditions: Dirichlet fixes u, Neumann fixes du/dx via a ghost cell.
Validate against the analytic solution of a decaying sine mode: u = exp(-alpha pi^2 t / L^2) sin(pi x / L).
""",
        },
        "Low_Latency_Cpp": {
            "low_latency_cpp_lock_free_ring_buffer.md": """# Lock-free single-producer single-consumer ring buffer

Capacity is a power of two so the index wraps with a mask: idx & (N - 1).
Producer owns head, consumer owns tail; each reads the other's index with memory_order_acquire and publishes its own with memory_order_release.
Place head and tail on separate cache lines (alignas(64)) to avoid false sharing.
push: if (head - tail == N) return false; buf[head & mask] = x; head.store(head + 1, release)
pop:  if (head == tail) return false; x = buf[tail & mask]; tail.store(tail + 1, release)
No allocations and no syscalls on the hot path. Pin both threads to isolated cores.
""",
            "low_latency_cpp_cache_line_false_sharing.txt": """Cache lines and false sharing

A cache line is 64 bytes on x86. Two threads writing different variables on the same line bounce it between cores (false sharing)
and can slow a loop 10x. Fix with alignas(64) or padding. std::hardware_destructive_interference_size gives the portable constant.
Other latency rules: prefer arrays of structs of hot fields only, avoid virtual calls in the hot loop, use reserve() to prevent
reallocation, prefetch with __builtin_prefetch for predictable streams, measure with perf stat (cache-misses, branch-misses) before optimising.
""",
        },
        "Cesium_Geospatial": {
            "cesium_geospatial_3d_tiles_streaming.md": """# Cesium 3D Tiles streaming

A tileset.json describes a spatial hierarchy: each tile has a bounding volume (region, box or sphere), a geometricError and content (b3dm, glb, pnts).
The renderer refines a tile when its screen-space error = geometricError * screenHeight / (distance * 2 * tan(fov/2)) exceeds maximumScreenSpaceError (default 16).
Tuning for weak GPUs: raise maximumScreenSpaceError to 24 to 32, limit tilesetCache size, enable foveated loading and skipLevelOfDetail.
Point clouds from ToF_3D_Mapping can be converted to .pnts or glTF tiles with a voxel octree.
""",
            "cesium_geospatial_ecef_wgs84_coordinate_transforms.txt": """ECEF and WGS84 coordinate transforms

WGS84: semi-major axis a = 6378137.0 m, flattening f = 1 / 298.257223563, e^2 = 2f - f^2.
Geodetic (lat, lon, h) to ECEF: N = a / sqrt(1 - e^2 sin^2 lat);
X = (N + h) cos lat cos lon; Y = (N + h) cos lat sin lon; Z = (N (1 - e^2) + h) sin lat.
For local robot work convert ECEF to ENU (east, north, up) around a reference point with a rotation matrix built from lat and lon.
Cesium uses Cartesian3 in ECEF meters; heights in glTF tiles are relative to the tile transform.
""",
        },
        "Industrial_PLC": {
            "industrial_plc_ladder_logic_scan_cycle.md": """# PLC ladder logic and the scan cycle

Scan cycle: read inputs into the image table -> execute the program top to bottom -> write outputs -> communications and housekeeping.
Typical scan time is 1 to 20 ms; a watchdog faults the CPU when the scan exceeds its limit.
Seal-in (latch) rung: start button parallel with the motor contact, in series with a normally closed stop, so one pulse keeps the motor on.
Use TON timers for debounce (50 ms) and one-shot (rising edge) instructions for counters.
Safety functions belong in a certified safety PLC or relay, never only in standard logic.
""",
            "industrial_plc_modbus_tcp_register_map.txt": """Modbus TCP register map

TCP port 502, unit id 1. Function codes: 03 read holding registers, 04 read input registers, 06 write single register, 16 write multiple registers.
Holding registers (16-bit): 40001 setpoint speed (rpm), 40002 start/stop command, 40003 mode, 40010 watchdog heartbeat from Jarvis.
Input registers: 30001 actual speed, 30002 motor current x10 (A), 30003 temperature x10 (C), 30004 fault word.
32-bit floats use two registers in big-endian word order. Poll at 100 ms; the PLC drops to safe mode if the heartbeat stops for 2 s.
""",
        },
    }


def ensure_ssd_folders():
    seeds = None
    created = []
    for node in MEMORY_NODES:
        folder = node["ssd"]
        if os.path.isdir(folder):
            continue
        os.makedirs(folder, exist_ok=True)
        if seeds is None:
            seeds = _seed_notes()
        for fname, text in seeds.get(node["id"], {"README.md": "# %s\n\nAdd notes for this node here.\n" % node["id"]}).items():
            with open(os.path.join(folder, fname), "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        created.append(node["id"])
    return created


# =====================================================================================
#  Lazy SSD readers
# =====================================================================================
TEXT_EXTS = {".md", ".txt", ".log", ".rst", ".json", ".yaml", ".yml", ".csv", ".ini", ".cfg", ".toml",
             ".py", ".c", ".cpp", ".h", ".hpp", ".js", ".ts", ".tex", ".xml", ".html", ".st", ".ld"}


def _is_text(name):
    return os.path.splitext(name)[1].lower() in TEXT_EXTS


def _read_head(path, nbytes):
    try:
        with open(path, "rb") as fh:
            raw = fh.read(nbytes)
    except OSError:
        return None
    if b"\x00" in raw:
        return None
    return raw.decode("utf-8", errors="replace")


def _snippet(text, terms, width=200):
    flat = re.sub(r"\s+", " ", text).strip()
    low = flat.lower()
    pos = -1
    for t in terms:
        p = low.find(t)
        if p != -1 and (pos == -1 or p < pos):
            pos = p
    if pos == -1:
        return flat[:width] + ("..." if len(flat) > width else "")
    start = max(0, pos - width // 3)
    piece = flat[start:start + width]
    return ("..." if start > 0 else "") + piece + ("..." if start + width < len(flat) else "")


def folder_files(folder, terms=(), limit=40, read_bytes=8192, preview_chars=220):
    """Lazy-load file list + previews from one SSD folder (only called on demand)."""
    out = []
    try:
        entries = sorted((e for e in os.scandir(folder) if e.is_file()), key=lambda e: e.name)
    except OSError:
        return out
    for e in entries[:limit]:
        try:
            size = e.stat().st_size
        except OSError:
            size = 0
        text = _read_head(e.path, read_bytes) if _is_text(e.name) else None
        out.append({"name": e.name, "size": size,
                    "snippet": _snippet(text, terms, preview_chars) if text else ""})
    return out


def _rss_mb():
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/self/statm") as fh:
                return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576.0
        if sys.platform == "win32":
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            psapi = ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            if psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return c.WorkingSetSize / 1048576.0
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            return peak / (1048576.0 if sys.platform == "darwin" else 1024.0)
    except Exception:
        pass
    return None


# =====================================================================================
#  Startup: folders, graph, engine, layout  (metadata only stays in RAM)
# =====================================================================================
NODE_BY_ID = {n["id"]: n for n in MEMORY_NODES}
_created_folders = ensure_ssd_folders()

GRAPH = nx.Graph()
for _n in MEMORY_NODES:
    GRAPH.add_node(_n["id"], subject=_n["subject"])
for _a, _b in MEMORY_LINKS:
    if _a in NODE_BY_ID and _b in NODE_BY_ID:
        GRAPH.add_edge(_a, _b)

ENGINE = ResonanceIndex3D(
    ids=[n["id"] for n in MEMORY_NODES],
    subjects={n["id"]: n["subject"] for n in MEMORY_NODES},
    folders={n["id"]: n["ssd"] for n in MEMORY_NODES},
    graph=GRAPH,
)
POSITIONS = compute_layout()
NEIGHBORS = {nid: sorted(GRAPH.neighbors(nid)) for nid in GRAPH.nodes}
LINKS_OUT = [{"source": a, "target": b} for a, b in GRAPH.edges]


def _display_path(node):
    rel = os.path.relpath(node["ssd"], BASE_DIR).replace(os.sep, "/")
    return "./" + rel + "/"


def _file_count(folder):
    try:
        return sum(1 for e in os.scandir(folder) if e.is_file())
    except OSError:
        return 0


# =====================================================================================
#  REST API
# =====================================================================================
app = Flask(__name__, static_folder=None)
app.config["JSON_SORT_KEYS"] = False


@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE_DIR, "templates"), "index.html", max_age=0)


@app.route("/api/brain")
def api_brain():
    nodes = []
    for n in MEMORY_NODES:
        x, y, z = POSITIONS[n["id"]]
        nodes.append({"id": n["id"], "subject": n["subject"],
                      "x": round(x, 4), "y": round(y, 4), "z": round(z, 4),
                      "ssd_path": _display_path(n), "file_count": _file_count(n["ssd"])})
    return jsonify({"nodes": nodes, "links": LINKS_OUT, "subjects": SUBJECT_COLORS, "shell": brain_shell()})


@app.route("/api/search", methods=["POST"])
def api_search():
    body = request.get_json(silent=True) or {}
    query = str(body.get("query", ""))[:200]

    ranked, stats = ENGINE.search(query)          # O(1) retrieval - timing measured inside

    t_load = time.perf_counter()                  # lazy SSD load of matched nodes only
    terms = [t for t in _tokens(query) if len(t) >= 2]
    matches = []
    for nid, score in ranked:
        node = NODE_BY_ID[nid]
        files = folder_files(node["ssd"], terms, limit=12)
        matches.append({"id": nid, "subject": node["subject"], "score": round(score, 6),
                        "ssd_path": _display_path(node), "file_count": len(files), "files": files})
    load_ms = (time.perf_counter() - t_load) * 1000.0

    return jsonify({"query": query, "query_ms": round(stats["query_ms"], 4), "load_ms": round(load_ms, 3),
                    "candidates": stats["candidates"], "probes": stats["probes"],
                    "total_nodes": len(MEMORY_NODES), "matches": matches})


@app.route("/api/node/<node_id>")
def api_node(node_id):
    node = NODE_BY_ID.get(node_id)
    if node is None:
        abort(404)
    files = folder_files(node["ssd"], (), limit=200, read_bytes=4096, preview_chars=400)
    return jsonify({"id": node_id, "subject": node["subject"], "ssd_path": _display_path(node),
                    "neighbors": [{"id": nb, "subject": NODE_BY_ID[nb]["subject"]} for nb in NEIGHBORS[node_id]],
                    "file_count": len(files),
                    "files": [{"name": f["name"], "size": f["size"], "preview": f["snippet"]} for f in files]})


@app.route("/api/node/<node_id>/file")
def api_node_file(node_id):
    node = NODE_BY_ID.get(node_id)
    if node is None:
        abort(404)
    name = os.path.basename(request.args.get("name", ""))
    path = os.path.join(node["ssd"], name)
    if not name or not os.path.isfile(path):
        abort(404)
    limit = 131072
    size = os.path.getsize(path)
    text = _read_head(path, limit) if _is_text(name) else None
    return jsonify({"name": name, "size": size, "truncated": size > limit, "content": text})


@app.route("/api/node/<node_id>/open", methods=["POST"])
def api_node_open(node_id):
    node = NODE_BY_ID.get(node_id)
    if node is None:
        abort(404)
    folder = os.path.abspath(node["ssd"])
    os.makedirs(folder, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(folder)  # noqa: S606  (local, trusted path from static config)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            subprocess.Popen(["xdg-open", folder], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return jsonify({"ok": False, "path": folder, "error": str(exc)}), 500
    return jsonify({"ok": True, "path": folder})


@app.route("/api/stats")
def api_stats():
    rss = _rss_mb()
    return jsonify({"rss_mb": None if rss is None else round(rss, 1), "nodes": len(MEMORY_NODES),
                    "links": len(LINKS_OUT), "engine": ENGINE.bucket_stats()})


if __name__ == "__main__":
    if _created_folders:
        print("Generated SSD folders with starter notes for: " + ", ".join(_created_folders))
    rss = _rss_mb()
    print("Jarvis 3D Brain  |  %d nodes, %d synapses  |  RSS %s MB" %
          (len(MEMORY_NODES), len(LINKS_OUT), "n/a" if rss is None else "%.1f" % rss))
    print("Open http://%s:%d" % (HOST, PORT))
    app.run(host=HOST, port=PORT, debug=False, threaded=True, use_reloader=False)
