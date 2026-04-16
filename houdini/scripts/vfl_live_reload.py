"""Houdini Python SOP — live-reload a .vfl file into a Wrangle on every cook.

Setup:
    1. Drop a Python SOP into your network (e.g. /obj/geo1/vfl_loader).
    2. Add two spare parameters to the Python SOP:
         - File parameter:           name = "vfl_path"
         - Node reference parameter: name = "target_wrangle"
       (Gear menu -> Edit Parameter Interface -> drag from Create Parameters)
    3. Set "vfl_path" to your .vfl file, e.g.:
           $HIP/../../houdini/vex/import_attributes.vfl
       or an absolute path.
    4. Set "target_wrangle" to the Attribute Wrangle's path, e.g.:
           ../attribwrangle1
       (relative paths are resolved from the Python SOP's parent).
    5. Paste this script into the Python SOP's Code field, or source it:
           exec(open("C:/.../houdini/scripts/vfl_live_reload.py").read())
    6. Wire the Python SOP upstream of (or alongside) the Wrangle so it
       cooks before the Wrangle evaluates.

On every cook the Python SOP reads the .vfl file and writes its contents
into the target Wrangle's "snippet" parameter.  Edit the .vfl in any
text editor, force-cook (Ctrl+Enter on the Python SOP or just scrub the
timeline), and the Wrangle picks up the changes.

The Python SOP passes its input geometry through unchanged.
"""

import hou


def main():
    node = hou.pwd()
    geo = node.geometry()

    # Pass input geometry through unchanged
    if node.inputs() and node.inputs()[0]:
        geo.merge(node.inputs()[0].geometry())

    # ── Read parameters ────────────────────────────────────
    vfl_parm = node.parm("vfl_path")
    wrangle_parm = node.parm("target_wrangle")

    if vfl_parm is None or wrangle_parm is None:
        raise hou.NodeError(
            "Add spare parameters 'vfl_path' (File) and "
            "'target_wrangle' (Node Reference) to this node.")

    vfl_path = vfl_parm.eval()
    wrangle_path = wrangle_parm.eval()

    if not vfl_path:
        raise hou.NodeError("Set the 'vfl_path' parameter to a .vfl file.")
    if not wrangle_path:
        raise hou.NodeError("Set the 'target_wrangle' parameter to a Wrangle node.")

    # ── Resolve wrangle node ───────────────────────────────
    wrangle = node.node(wrangle_path)
    if wrangle is None:
        # Try as absolute path
        wrangle = hou.node(wrangle_path)
    if wrangle is None:
        raise hou.NodeError("Cannot find node: {}".format(wrangle_path))

    snippet = wrangle.parm("snippet")
    if snippet is None:
        raise hou.NodeError("Node {} has no 'snippet' parameter — "
                            "is it an Attribute Wrangle?".format(wrangle.path()))

    # ── Read .vfl and push into wrangle ────────────────────
    with open(vfl_path, "r") as f:
        contents = f.read()

    snippet.set(contents)


main()
