"""Houdini Python SOP — import organism data from a pivotal moment snapshot.

Usage:
    1. Add a Python SOP to your network.
    2. Set the 'json_path' string parameter on the node to the path of a
       pivotal_moments.json event entry (or a standalone snapshot JSON).
    3. Paste this script into the Python SOP's Code field, or source it
       via: exec(open("<path>/import_organism_json.py").read())

The script reads the first event in the file (or the top-level object if
it contains an "organisms" key directly) and creates one point per
organism with full attribute coverage.
"""

import json
import hou

node = hou.pwd()
geo = node.geometry()
geo.clear()

# ── Read JSON ───────────────────────────────────────────
json_path = node.parm("json_path").eval()
if not json_path:
    raise hou.NodeError("Set the 'json_path' parameter to a snapshot JSON path.")

with open(json_path, "r") as f:
    data = json.load(f)

# Navigate to the organism list — handle both full-index and single-event files.
if "events" in data and len(data["events"]) > 0:
    event = data["events"][0]
elif "organisms" in data:
    event = data
else:
    raise hou.NodeError("JSON has no 'events' or 'organisms' key.")

organisms = event.get("organisms", [])
if not organisms:
    return

# ── Module names (must match constants.py) ──────────────
MODULE_NAMES = [
    "PHOTO", "CHEMO", "CONSUME", "MOVE", "FORAGE",
    "DEFENSE", "DETOX", "TOXPROD", "VRESIST", "SOCIAL", "MEDIATE",
]

# Trophic role lookup (mirrors StatsMixin._trophic_roles logic)
ROLE_NAMES = {0: "producer", 1: "herbivore", 2: "carnivore",
              3: "detritivore", 4: "omnivore"}

# ── Create point attributes ────────────────────────────
# Integers
for name in ("id", "age", "generation", "parent_id", "merger_count",
             "cascade_phase", "transfer_count", "n_modules", "is_mobile",
             "is_mature", "role_id"):
    geo.addAttrib(hou.attribType.Point, name, 0)

# Floats
for name in ("energy", "genomic_stress", "viral_load", "lysogenic_strength",
             "immune_experience", "relationship_score", "hijack_intensity",
             "dev_progress", "energy_norm", "age_norm", "stress_norm"):
    geo.addAttrib(hou.attribType.Point, name, 0.0)

# Strings
for name in ("modules", "trophic_role"):
    geo.addAttrib(hou.attribType.Point, name, "")

# Per-module presence: i@has_PHOTO, i@has_CHEMO, ...
for mod in MODULE_NAMES:
    geo.addAttrib(hou.attribType.Point, "has_" + mod, 0)

# Per-module active: i@active_PHOTO, ...
for mod in MODULE_NAMES:
    geo.addAttrib(hou.attribType.Point, "active_" + mod, 0)

# Per-module usage: f@usage_PHOTO, ...
for mod in MODULE_NAMES:
    geo.addAttrib(hou.attribType.Point, "usage_" + mod, 0.0)

# ── Flatten weight names from the first organism to discover keys ──
_sample_weights = organisms[0].get("weights", {})
_weight_attrs = []  # (attr_name, module_key, param_key)
for mod_key, params in _sample_weights.items():
    if isinstance(params, dict):
        for param_key in params:
            attr_name = "w_" + param_key
            _weight_attrs.append((attr_name, mod_key, param_key))
            geo.addAttrib(hou.attribType.Point, attr_name, 0.0)

# ── Create points ──────────────────────────────────────
for org in organisms:
    pt = geo.createPoint()

    row = org.get("row", 0)
    col = org.get("col", 0)
    pt.setPosition(hou.Vector3(col * 0.1, 0, row * 0.1))

    # Scalar integers
    pt.setAttribValue("id", org.get("id", 0))
    pt.setAttribValue("age", org.get("age", 0))
    pt.setAttribValue("generation", org.get("generation", 0))
    pt.setAttribValue("parent_id", org.get("parent_ids", 0))
    pt.setAttribValue("merger_count", org.get("merger_count", 0))
    pt.setAttribValue("cascade_phase", org.get("genomic_cascade_phase", 0))
    pt.setAttribValue("transfer_count", org.get("transfer_count", 0))
    pt.setAttribValue("is_mature", int(org.get("is_mature", False)))

    # Scalar floats
    energy = org.get("energy", 0.0)
    age = org.get("age", 0)
    genomic_stress = org.get("genomic_stress", 0.0)

    pt.setAttribValue("energy", energy)
    pt.setAttribValue("genomic_stress", genomic_stress)
    pt.setAttribValue("viral_load", org.get("viral_load", 0.0))
    pt.setAttribValue("lysogenic_strength", org.get("lysogenic_strength", 0.0))
    pt.setAttribValue("immune_experience", org.get("immune_experience", 0.0))
    pt.setAttribValue("relationship_score", org.get("relationship_score", 0.0))
    pt.setAttribValue("hijack_intensity", org.get("hijack_intensity", 0.0))
    pt.setAttribValue("dev_progress", org.get("dev_progress", 0.0))

    # Normalised values
    pt.setAttribValue("energy_norm", energy / 150.0)
    pt.setAttribValue("age_norm", age / 200.0)
    pt.setAttribValue("stress_norm", genomic_stress / 15.0)

    # Module presence / active / usage
    mod_present = org.get("module_present", {})
    mod_active = org.get("module_active", {})
    mod_usage = org.get("module_usage", {})
    present_list = []
    n_modules = 0
    is_mobile = 0
    for mod in MODULE_NAMES:
        has = int(mod_present.get(mod, False))
        pt.setAttribValue("has_" + mod, has)
        pt.setAttribValue("active_" + mod, int(mod_active.get(mod, False)))
        pt.setAttribValue("usage_" + mod, mod_usage.get(mod, 0.0))
        if has:
            present_list.append(mod)
            n_modules += 1
        if mod == "MOVE" and mod_active.get(mod, False):
            is_mobile = 1

    pt.setAttribValue("modules", " ".join(present_list))
    pt.setAttribValue("n_modules", n_modules)
    pt.setAttribValue("is_mobile", is_mobile)

    # Trophic role — simple heuristic matching the simulation's classifier
    has_consume = mod_present.get("CONSUME", False)
    if has_consume:
        weights = org.get("weights", {})
        consume_w = weights.get("CONSUME", {})
        prey_sel = consume_w.get("prey_selectivity", 0.0)
        decomp_pref = consume_w.get("decomp_preference", 0.0)
        # Sigmoid transform (matches simulation)
        import math
        ps = 1.0 / (1.0 + math.exp(-prey_sel))
        dp = 1.0 / (1.0 + math.exp(-decomp_pref))
        if dp >= 0.6:
            role_id = 3  # detritivore
        elif ps < 0.35:
            role_id = 1  # herbivore
        elif ps > 0.65:
            role_id = 2  # carnivore
        else:
            role_id = 4  # omnivore
    else:
        role_id = 0  # producer

    pt.setAttribValue("role_id", role_id)
    pt.setAttribValue("trophic_role", ROLE_NAMES.get(role_id, "unknown"))

    # Flattened weight attributes
    weights = org.get("weights", {})
    for attr_name, mod_key, param_key in _weight_attrs:
        mod_dict = weights.get(mod_key, {})
        if isinstance(mod_dict, dict):
            pt.setAttribValue(attr_name, mod_dict.get(param_key, 0.0))
