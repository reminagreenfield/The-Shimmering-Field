"""Sync a .vfl file's contents into an Attribute Wrangle's VEXpression field.

Standalone function — call from a shelf tool, Python Shell, or callback:

    import sys
    sys.path.append("C:/path/to/The-Shimmering-Field/houdini/scripts")
    from sync_vex_from_file import sync_vex_from_file

    sync_vex_from_file("/obj/geo1/attribwrangle1",
                       "C:/path/to/The-Shimmering-Field/houdini/vex/import_attributes.vfl")

Or as a one-liner in the Houdini Python Shell:

    hou.node("/obj/geo1/attribwrangle1").parm("snippet").set(
        open("C:/.../houdini/vex/import_attributes.vfl").read())

Parameters:
    wrangle_path : str  — absolute Houdini node path to an Attribute Wrangle
    vfl_path     : str  — filesystem path to a .vfl file
"""

import hou


def sync_vex_from_file(wrangle_path, vfl_path):
    """Read *vfl_path* and write its contents into the wrangle's snippet parm.

    Raises hou.NodeError on bad node path, missing parm, or file read failure.
    """
    wrangle = hou.node(wrangle_path)
    if wrangle is None:
        raise hou.NodeError("Node not found: {}".format(wrangle_path))

    parm = wrangle.parm("snippet")
    if parm is None:
        raise hou.NodeError("Node {} has no 'snippet' parameter — "
                            "is it an Attribute Wrangle?".format(wrangle_path))

    with open(vfl_path, "r") as f:
        contents = f.read()

    parm.set(contents)
