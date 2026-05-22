from __future__ import annotations

import argparse
import collections
import re
import time
from pathlib import Path
from typing import OrderedDict

import trimesh
import viser


MESH_EXTENSIONS = {".obj", ".ply", ".stl", ".off", ".glb", ".gltf"}


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def collect_mesh_paths(mesh_dir: Path) -> list[Path]:
    mesh_paths = [path for path in mesh_dir.iterdir() if path.suffix.lower() in MESH_EXTENSIONS]
    mesh_paths.sort(key=natural_sort_key)
    return mesh_paths


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh", process=False, maintain_order=True)

    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            raise ValueError(f"No geometry found in {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Unsupported mesh type for {mesh_path}: {type(mesh)!r}")

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError(f"Mesh has no faces: {mesh_path}")

    return mesh


class MeshCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max(1, max_items)
        self._cache: OrderedDict[Path, trimesh.Trimesh] = collections.OrderedDict()

    def get(self, mesh_path: Path) -> trimesh.Trimesh:
        cached = self._cache.get(mesh_path)
        if cached is not None:
            self._cache.move_to_end(mesh_path)
            return cached

        mesh = load_mesh(mesh_path)
        self._cache[mesh_path] = mesh
        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return mesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a mesh sequence with viser.")
    parser.add_argument(
        "mesh_dir",
        nargs="?",
        default="test/debug_viz_output",
        help="Directory containing the mesh sequence.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host for the viser server.")
    parser.add_argument("--port", type=int, default=8080, help="Port for the viser server.")
    parser.add_argument(
        "--cache-size",
        type=int,
        default=8,
        help="Maximum number of meshes cached in memory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_dir = Path(args.mesh_dir).expanduser().resolve()

    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory does not exist: {mesh_dir}")

    mesh_paths = collect_mesh_paths(mesh_dir)
    if not mesh_paths:
        raise FileNotFoundError(f"No mesh files found in {mesh_dir}")

    server = viser.ViserServer(host=args.host, port=args.port)
    cache = MeshCache(max_items=args.cache_size)

    frame_slider = server.gui.add_slider(
        "frame",
        min=0,
        max=len(mesh_paths) - 1,
        step=1,
        initial_value=0,
        hint="Slide to switch the visible mesh frame.",
    )

    current_handle: list[object | None] = [None]
    current_index: list[int] = [0]

    def show_frame(frame_index: int) -> None:
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= len(mesh_paths):
            return

        mesh_path = mesh_paths[frame_index]
        mesh = cache.get(mesh_path)

        if current_handle[0] is not None:
            current_handle[0].remove()

        current_handle[0] = server.scene.add_mesh_trimesh(
            name="/mesh_sequence/current",
            mesh=mesh,
            visible=True,
        )
        current_index[0] = frame_index
        print(f"Showing frame {frame_index}: {mesh_path.name}")

    @frame_slider.on_update
    def _(_event: viser.GuiEvent) -> None:
        next_index = int(frame_slider.value)
        if next_index != current_index[0]:
            show_frame(next_index)

    show_frame(0)
    print(f"Loaded {len(mesh_paths)} mesh frames from {mesh_dir}")
    print(f"Open http://{args.host}:{args.port} in your browser")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()