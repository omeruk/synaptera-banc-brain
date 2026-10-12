"""The only place that touches FlyGym's private API (review M6).

v1 reached into ``Fly._update_vision``, ``_last_vision_update_time``, ``_curr_visual_input`` and
``_floor_contacts`` from flight/body.py. They are used here, once, behind three functions; the FlyGym version is
pinned (1.2.1) and ``tests/test_flygym_private.py`` fails loudly on an upgrade that moves them.
"""
import numpy as np

REQUIRED_FLYGYM = "1.2.1"


def render_eyes(fly, sim) -> np.ndarray:
    """Render both compound eyes now; (2, 721, 2) ommatidia readout. Needs enable_vision=True."""
    fly._last_vision_update_time = -np.inf
    fly._update_vision(sim)
    return fly._curr_visual_input


def floor_contact_items(fly) -> list:
    return list(fly._floor_contacts.items())


def drop_floor_contact(fly, key) -> None:
    fly._floor_contacts[key].remove()
    del fly._floor_contacts[key]


def check_private_api(fly_cls) -> list:
    """Names of the private attributes this adapter needs that the class no longer provides."""
    need = ("_update_vision",)
    return [n for n in need if not hasattr(fly_cls, n)]
