"""Interactive viewer for predefined contact points.

Renders three meshes side-by-side in an Open3D window:

    [ club mesh ]      [ right MANO ]      [ left MANO ]

Each entry in the JSON config is shown as a colored sphere with a thin line
to its partner vertex.

Operations
----------
* **Add a contact point** — Shift + left-click two vertices in the window.
  The pair is auto-classified by which meshes they hit:
    right hand + club  -> right_hand_object_contacts
    left  hand + club  -> left_hand_object_contacts
    right hand + left  -> hand_hand_contacts
  The new entry is appended to the JSON, saved, and the view refreshes.

* **Terminal commands** (type in the launching terminal + Enter):
    r  -> re-read the JSON from disk (e.g. after manual edits)
    c  -> clear the current picks without writing
    q  -> quit

(Open3D's vertex picker doesn't support keyboard shortcuts, hence the
terminal commands.)

Usage:
    python utils/vis_predefined_contacts.py config/baseball_golf.json
"""
from __future__ import annotations

import argparse
import colorsys
import json
import os
import pickle
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh


_HERE = Path(__file__).resolve().parent
_PROJECT = _HERE.parent
_MANO_DIR = _PROJECT / "MANO"

# Layout offsets (meters) for the 3 reference meshes. Right hand is at origin.
LAYOUT = {
    "club":  np.array([-0.50, 0.0, 0.0], dtype=np.float64),
    "right": np.array([ 0.00, 0.0, 0.0], dtype=np.float64),
    "left":  np.array([ 0.30, 0.0, 0.0], dtype=np.float64),
}

# Base mesh colors (RGB 0..1) — flat shading, contact spheres go on top.
COLOR_RIGHT = (0.78, 0.92, 0.78)
COLOR_LEFT  = (0.80, 0.86, 0.96)
COLOR_CLUB  = (0.95, 0.85, 0.65)

SPHERE_R = 0.004   # 4 mm
LINE_W   = 2.0

# Distinct visuals for hand_attract_tips (force-closure attract anchors).
TIP_SPHERE_R = 0.006   # 6 mm — slightly larger so they stand out
TIP_COLOR    = (1.0, 0.85, 0.10)   # gold (RGB 0..1)


# --------------------------------------------------------------------- mesh ---
def _load_mano_template(side: str):
    """Return (verts (778,3), faces (1538,3)) for the chosen MANO hand."""
    name = "MANO_RIGHT.pkl" if side == "right" else "MANO_LEFT.pkl"
    p = _MANO_DIR / name
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    verts = np.asarray(d["v_template"], dtype=np.float64)
    faces = np.asarray(d["f"], dtype=np.int64)
    return verts, faces


def _build_o3d_mesh(verts: np.ndarray, faces: np.ndarray, color, offset):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(verts + offset)
    m.triangles = o3d.utility.Vector3iVector(faces)
    m.compute_vertex_normals()
    m.paint_uniform_color(color)
    return m


def _load_club_verts_faces(path: Optional[str]):
    if not path:
        return None, None
    p = Path(path).expanduser()
    if not p.is_file():
        print(f"[vis] club_mesh_path 不存在: {p}", file=sys.stderr)
        return None, None
    mesh = trimesh.load(str(p), force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
    return (np.asarray(mesh.vertices, dtype=np.float64),
            np.asarray(mesh.faces, dtype=np.int64))


# ---------------------------------------------------------------- color util ---
def _palette(n: int) -> List[Tuple[float, float, float]]:
    """Distinct HSV-spaced colors."""
    out = []
    for i in range(max(1, n)):
        h = (i / max(1, n)) % 1.0
        out.append(colorsys.hsv_to_rgb(h, 0.85, 0.95))
    return out


# ----------------------------------------------------------- contact geometry ---
def _make_sphere(center, color, radius=SPHERE_R):
    s = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=12)
    s.translate(np.asarray(center, dtype=np.float64))
    s.paint_uniform_color(np.asarray(color, dtype=np.float64))
    s.compute_vertex_normals()
    return s


def _make_cylinder_between(p1, p2, color, radius=0.0008):
    """Thin cylinder along p1->p2 (used as a 'line' since LineSet can't be merged
    into a TriangleMesh and VisualizerWithVertexSelection only takes one)."""
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    h = float(np.linalg.norm(p2 - p1))
    if h < 1e-6:
        return None
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=h, resolution=8)
    z = np.array([0.0, 0.0, 1.0])
    v = (p2 - p1) / h
    axis = np.cross(z, v)
    a = float(np.linalg.norm(axis))
    if a < 1e-8:
        if v[2] < 0:
            cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
                [np.pi, 0, 0]), center=(0, 0, 0))
    else:
        ang = float(np.arccos(np.clip(float(np.dot(z, v)), -1.0, 1.0)))
        cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(
            axis / a * ang), center=(0, 0, 0))
    cyl.translate((p1 + p2) / 2.0)
    cyl.paint_uniform_color(np.asarray(color, dtype=np.float64))
    cyl.compute_vertex_normals()
    return cyl


def _combine_meshes(meshes: List[o3d.geometry.TriangleMesh]) -> o3d.geometry.TriangleMesh:
    """Concatenate a list of TriangleMesh into one, preserving per-vertex colors."""
    out = o3d.geometry.TriangleMesh()
    all_v, all_t, all_c = [], [], []
    offset = 0
    for m in meshes:
        if m is None:
            continue
        v = np.asarray(m.vertices, dtype=np.float64)
        t = np.asarray(m.triangles, dtype=np.int32)
        if len(v) == 0 or len(t) == 0:
            continue
        if m.has_vertex_colors() and len(np.asarray(m.vertex_colors)):
            c = np.asarray(m.vertex_colors, dtype=np.float64)
        else:
            c = np.tile(np.asarray([0.7, 0.7, 0.7]), (len(v), 1))
        all_v.append(v); all_t.append(t + offset); all_c.append(c)
        offset += len(v)
    if not all_v:
        return out
    out.vertices = o3d.utility.Vector3dVector(np.vstack(all_v))
    out.triangles = o3d.utility.Vector3iVector(np.vstack(all_t))
    out.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_c))
    out.compute_vertex_normals()
    return out


def _vid_pos(verts_world: np.ndarray, vid: int, label: str):
    """verts_world 必须已经是 world 坐标（含 LAYOUT 偏移）。"""
    if vid < 0 or vid >= len(verts_world):
        print(f"[vis] {label}: vid={vid} 超出范围 0..{len(verts_world)-1}，跳过",
              file=sys.stderr)
        return None
    return verts_world[vid]


def build_dynamic_geometries(
    cfg: dict,
    right_verts: np.ndarray,
    left_verts: np.ndarray,
    club_verts: Optional[np.ndarray],
) -> List[o3d.geometry.Geometry]:
    """Build all spheres + lines from a config dict."""
    rh = cfg.get("right_hand_object_contacts", []) or []
    lh = cfg.get("left_hand_object_contacts", []) or []
    hh = cfg.get("hand_hand_contacts", []) or []
    n_total = len(rh) + len(lh) + len(hh)
    palette = _palette(n_total)
    geoms: List[o3d.geometry.Geometry] = []

    print(f"\n=== contacts loaded ({n_total}) ===")
    ci = 0

    def add_pair(a, b, color, label):
        if a is None or b is None:
            return
        geoms.append(_make_sphere(a, color))
        geoms.append(_make_sphere(b, color))
        cyl = _make_cylinder_between(a, b, color)
        if cyl is not None:
            geoms.append(cyl)
        print(f"  {label}  rgb=({color[0]:.2f},{color[1]:.2f},{color[2]:.2f})")

    # right hand <-> club
    for c in rh:
        col = palette[ci]; ci += 1
        name = c.get("name", f"R-O-{ci}")
        mvid = int(c.get("mano_vid", -1))
        ovid = int(c.get("obj_vid", -1))
        ph = _vid_pos(right_verts, mvid, f"[R-O] {name}.mano_vid")
        if club_verts is None:
            print(f"[vis] [R-O] {name}: 无 club mesh，仅画 hand 端")
            if ph is not None:
                geoms.append(_make_sphere(ph, col))
            continue
        po = _vid_pos(club_verts, ovid, f"[R-O] {name}.obj_vid")
        add_pair(ph, po, col,
                 f"[R-O] {name}  mano_vid={mvid} obj_vid={ovid}")

    # left hand <-> club
    for c in lh:
        col = palette[ci]; ci += 1
        name = c.get("name", f"L-O-{ci}")
        mvid = int(c.get("mano_vid", -1))
        ovid = int(c.get("obj_vid", -1))
        ph = _vid_pos(left_verts, mvid, f"[L-O] {name}.mano_vid")
        if club_verts is None:
            if ph is not None:
                geoms.append(_make_sphere(ph, col))
            continue
        po = _vid_pos(club_verts, ovid, f"[L-O] {name}.obj_vid")
        add_pair(ph, po, col,
                 f"[L-O] {name}  mano_vid={mvid} obj_vid={ovid}")

    # right hand <-> left hand
    for c in hh:
        col = palette[ci]; ci += 1
        name = c.get("name", f"R-L-{ci}")
        rvid = int(c.get("right_mano_vid", -1))
        lvid = int(c.get("left_mano_vid", -1))
        pr = _vid_pos(right_verts, rvid, f"[R-L] {name}.right_mano_vid")
        pl = _vid_pos(left_verts,  lvid, f"[R-L] {name}.left_mano_vid")
        add_pair(pr, pl, col,
                 f"[R-L] {name}  right_vid={rvid} left_vid={lvid}")

    # hand_attract_tips: force-closure attract anchors (no partner — drawn as
    # standalone gold spheres on each hand).
    tips_section = cfg.get("hand_attract_tips") or {}
    for side, host_verts in (("right", right_verts), ("left", left_verts)):
        tips = tips_section.get(side) or []
        for vid_raw in tips:
            try:
                vid = int(vid_raw)
            except (TypeError, ValueError):
                print(f"[vis] [tip] {side}: 跳过非整数 vid={vid_raw!r}")
                continue
            p = _vid_pos(host_verts, vid, f"[tip] {side}.attract_tip")
            if p is None:
                continue
            geoms.append(_make_sphere(p, TIP_COLOR, radius=TIP_SPHERE_R))
            print(f"  [tip-{side[0].upper()}] vid={vid}  rgb=gold")

    print(f"=== drawn {len(geoms)} primitives ===\n")
    return geoms


# ----------------------------------------------------------- io / live reload ---
def load_config(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[vis] JSON 解析失败 (保留旧配置): {e}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"[vis] 配置文件不存在: {path}", file=sys.stderr)
        return None
    return cfg


# ------------------------------------------------ pick → contact persistence --
def _nearest_mesh_and_vid(coord: np.ndarray, meshes_world: Dict[str, np.ndarray]):
    """meshes_world: {"right": ..., "left": ..., "club": ...}.  Returns (key, vid, dist)."""
    best = None
    for key, verts in meshes_world.items():
        if verts is None:
            continue
        d = np.linalg.norm(verts - coord, axis=1)
        i = int(d.argmin())
        if best is None or d[i] < best[2]:
            best = (key, i, float(d[i]))
    return best


def _next_auto_name(cfg: dict, prefix: str) -> str:
    n = 0
    for section in ("right_hand_object_contacts",
                    "left_hand_object_contacts",
                    "hand_hand_contacts"):
        for entry in cfg.get(section, []) or []:
            name = entry.get("name", "")
            if name.startswith(f"auto_{prefix}_"):
                try:
                    n = max(n, int(name.split("_")[-1]) + 1)
                except ValueError:
                    pass
    return f"auto_{prefix}_{n}"


def _commit_pair_to_config(cfg_path: Path, p1: np.ndarray, p2: np.ndarray,
                           meshes_world: Dict[str, np.ndarray]) -> bool:
    """Classify two picked coords and append a new contact entry to JSON.
    Returns True iff the file was written."""
    c1 = _nearest_mesh_and_vid(p1, meshes_world)
    c2 = _nearest_mesh_and_vid(p2, meshes_world)
    if c1 is None or c2 is None:
        print("[vis] 无法分类拾取点 (mesh 缺失?)")
        return False
    m1, v1, d1 = c1
    m2, v2, d2 = c2
    print(f"[vis] pick #1 -> {m1} vid={v1} (snap dist={d1*1000:.1f}mm)")
    print(f"[vis] pick #2 -> {m2} vid={v2} (snap dist={d2*1000:.1f}mm)")

    if m1 == m2:
        print(f"[vis] ✗ 两个点都在 {m1} 上，不是合法的 contact pair")
        return False

    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[vis] ✗ 读取 JSON 失败: {e}")
        return False

    pair = {m1, m2}
    if pair == {"right", "club"}:
        mvid = v1 if m1 == "right" else v2
        ovid = v2 if m1 == "right" else v1
        section = "right_hand_object_contacts"
        entry = {"name": _next_auto_name(cfg, "R_O"),
                 "mano_vid": mvid, "obj_vid": ovid, "weight": 1.0}
    elif pair == {"left", "club"}:
        mvid = v1 if m1 == "left" else v2
        ovid = v2 if m1 == "left" else v1
        section = "left_hand_object_contacts"
        entry = {"name": _next_auto_name(cfg, "L_O"),
                 "mano_vid": mvid, "obj_vid": ovid, "weight": 1.0}
    elif pair == {"right", "left"}:
        rvid = v1 if m1 == "right" else v2
        lvid = v2 if m1 == "right" else v1
        section = "hand_hand_contacts"
        entry = {"name": _next_auto_name(cfg, "R_L"),
                 "right_mano_vid": rvid, "left_mano_vid": lvid, "weight": 1.0}
    else:
        print(f"[vis] ✗ 不支持的组合: {m1} + {m2}")
        return False

    cfg.setdefault(section, []).append(entry)
    tmp = cfg_path.with_suffix(cfg_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, cfg_path)
    print(f"[vis] ✓ 已写入 {section}: {entry}")
    return True


# --------------------------------------------------------------------- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config", help="路径，例如 config/baseball_golf.json")
    args = ap.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.is_file():
        print(f"[vis] 配置文件不存在: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    # ---- load static meshes ----
    rv, rf = _load_mano_template("right")
    lv, lf = _load_mano_template("left")
    right_mesh = _build_o3d_mesh(rv, rf, COLOR_RIGHT, LAYOUT["right"])
    left_mesh  = _build_o3d_mesh(lv, lf, COLOR_LEFT,  LAYOUT["left"])
    right_verts_world = rv + LAYOUT["right"]  # for sphere placement
    left_verts_world  = lv + LAYOUT["left"]

    cfg = load_config(cfg_path) or {}
    cv, cf = _load_club_verts_faces(cfg.get("club_mesh_path"))
    if cv is not None:
        club_mesh = _build_o3d_mesh(cv, cf, COLOR_CLUB, LAYOUT["club"])
        club_verts_world = cv + LAYOUT["club"]
    else:
        club_mesh = None
        club_verts_world = None

    # ---- visualizer ----
    # NOTE: VisualizerWithVertexSelection only accepts ONE geometry, so every
    # mesh + every contact marker has to be merged into a single TriangleMesh.
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=f"Predefined contacts: {cfg_path.name}",
                      width=1280, height=860)

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=[0, -0.15, 0])

    def _static_pieces():
        out = [right_mesh, left_mesh, coord]
        if club_mesh is not None:
            out.append(club_mesh)
        return out

    combined_holder = {"mesh": None, "added": False}
    first_add = {"flag": True}  # only first add resets bounding box

    def _set_combined(new_mesh):
        if combined_holder["added"] and combined_holder["mesh"] is not None:
            vis.remove_geometry(combined_holder["mesh"], reset_bounding_box=False)
        combined_holder["mesh"] = new_mesh
        ok = vis.add_geometry(new_mesh, reset_bounding_box=first_add["flag"])
        if not ok:
            print("[vis] ✗ add_geometry 失败 (合并 mesh 顶点数 = "
                  f"{len(np.asarray(new_mesh.vertices))})")
        combined_holder["added"] = True
        first_add["flag"] = False

    def rebuild():
        nonlocal club_mesh, club_verts_world
        new_cfg = load_config(cfg_path)
        if new_cfg is None:
            return
        # club mesh path can change at runtime
        new_path = new_cfg.get("club_mesh_path")
        cur_path = (cfg.get("club_mesh_path") if cfg else None)
        if new_path != cur_path:
            cv2_v, cf2 = _load_club_verts_faces(new_path)
            if cv2_v is not None:
                club_mesh = _build_o3d_mesh(cv2_v, cf2, COLOR_CLUB, LAYOUT["club"])
                club_verts_world = cv2_v + LAYOUT["club"]
            else:
                club_mesh = None
                club_verts_world = None

        dyn = build_dynamic_geometries(
            new_cfg, right_verts_world, left_verts_world, club_verts_world)
        combined = _combine_meshes(_static_pieces() + dyn)
        _set_combined(combined)
        cfg.clear(); cfg.update(new_cfg)

    rebuild()

    # ---- pick → auto-add ----
    # When the user has shift+clicked 2 vertices we classify and write to JSON.
    # The selection callback may fire while we are still mutating geometry, so
    # we just stash a flag and act in the main loop.
    pending_commit = {"flag": False}

    def on_selection_changed():
        picks = vis.get_picked_points()
        if len(picks) >= 2 and not pending_commit["flag"]:
            pending_commit["flag"] = True

    vis.register_selection_changed_callback(on_selection_changed)

    # ---- terminal command channel (Shift-pick is in-window; commands are stdin) ----
    cmd_q: "queue.Queue[str]" = queue.Queue()

    def _stdin_reader():
        for line in sys.stdin:
            cmd_q.put(line.strip().lower())

    threading.Thread(target=_stdin_reader, daemon=True).start()

    print(f"[vis] config = {cfg_path}")
    print("[vis] 操作:")
    print("[vis]   * 在窗口中 Shift + 左键 拾取顶点; 拾够 2 个会自动写入 config")
    print("[vis]       右手 + 球杆 -> right_hand_object_contacts")
    print("[vis]       左手 + 球杆 -> left_hand_object_contacts")
    print("[vis]       右手 + 左手 -> hand_hand_contacts")
    print("[vis]   * 终端命令:  r=reload   c=clear picks   q=quit")

    def _meshes_world():
        return {
            "right": right_verts_world,
            "left":  left_verts_world,
            "club":  club_verts_world,
        }

    try:
        while vis.poll_events():
            # 1) terminal commands
            try:
                while True:
                    cmd = cmd_q.get_nowait()
                    if cmd == "r":
                        print(f"[vis] reload {cfg_path.name}")
                        rebuild()
                    elif cmd == "c":
                        vis.clear_picked_points()
                        print("[vis] picks cleared")
                    elif cmd in ("q", "quit", "exit"):
                        print("[vis] quit")
                        return
                    elif cmd == "":
                        pass
                    else:
                        print(f"[vis] 未知命令: {cmd!r}  (支持 r/c/q)")
            except queue.Empty:
                pass

            # 2) deferred pick commit
            if pending_commit["flag"]:
                picks = vis.get_picked_points()
                if len(picks) >= 2:
                    p1 = np.asarray(picks[0].coord, dtype=np.float64)
                    p2 = np.asarray(picks[1].coord, dtype=np.float64)
                    ok = _commit_pair_to_config(cfg_path, p1, p2, _meshes_world())
                    vis.clear_picked_points()
                    if ok:
                        rebuild()
                pending_commit["flag"] = False

            vis.update_renderer()
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("[vis] 用户中断")
    finally:
        vis.destroy_window()


if __name__ == "__main__":
    main()
