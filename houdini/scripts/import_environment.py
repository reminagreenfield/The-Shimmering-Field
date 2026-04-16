"""Houdini Python SOP — import environment grids from a pivotal moment snapshot.

Usage:
    1. Add a Python SOP to your network.
    2. Set the 'json_path' string parameter on the node to the path of a
       pivotal_moments.json event entry (or a standalone snapshot JSON).
    3. Paste this script into the Python SOP's Code field.

Creates an 80x80 point grid with one point per cell, carrying all
environment layers as float attributes.  Point positions match the
organism import scale (col*0.1 -> X, row*0.1 -> Z).
"""

import json
import hou


def ensure_point_attrib(geo, attr_type, name, default):
    existing = geo.findPointAttrib(name)
    if existing is None:
        return geo.addAttrib(attr_type, name, default)
    return existing


def main():
    node = hou.pwd()
    geo = node.geometry()
    geo.clear()

    # ── Read JSON ───────────────────────────────────────────
    json_path = node.parm("json_path").eval()
    if not json_path:
        raise hou.NodeError("Set the 'json_path' parameter to a snapshot JSON path.")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Navigate to environment dict
    if "events" in data and len(data["events"]) > 0:
        event = data["events"][0]
    elif "environment" in data:
        event = data
    else:
        raise hou.NodeError("JSON has no 'events' or 'environment' key.")

    env = event.get("environment", {})
    if not env:
        raise hou.NodeError("No environment data in snapshot.")

    # ── Determine grid size from the first layer ────────────
    first_layer = next(iter(env.values()))
    grid_rows = len(first_layer)
    grid_cols = len(first_layer[0]) if grid_rows > 0 else 0

    # ── Create point attributes ────────────────────────────
    layer_names = {
        "toxic":                "toxicity",
        "nutrients":            "nutrients",
        "decomposition":        "decomposition",
        "fungal_density":       "fungal_density",
        "ecosystem_integrity":  "ecosystem_integrity",
        "viral_particles":      "viral_particles",
    }

    for attr_name in layer_names.values():
        ensure_point_attrib(geo, hou.attribType.Point, attr_name, 0.0)

    ensure_point_attrib(geo, hou.attribType.Point, "row", 0)
    ensure_point_attrib(geo, hou.attribType.Point, "col", 0)

    # ── Create grid points ─────────────────────────────────
    for row in range(grid_rows):
        for col in range(grid_cols):
            pt = geo.createPoint()
            pt.setPosition(hou.Vector3(col * 0.1, 0, row * 0.1))
            pt.setAttribValue("row", row)
            pt.setAttribValue("col", col)

            for json_key, attr_name in layer_names.items():
                layer = env.get(json_key)
                if layer is not None:
                    pt.setAttribValue(attr_name, float(layer[row][col]))


main()
