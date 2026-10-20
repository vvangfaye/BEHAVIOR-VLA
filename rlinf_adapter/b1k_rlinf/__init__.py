"""RLinf extension for the joint-subtask BEHAVIOR pi0.5 policy."""


def register():
    """Register in every RLinf driver / worker via RLINF_EXT_MODULE."""
    from rlinf.models import register_model

    from .factory import build_model

    register_model("openpi", build_model, category="embodied", force=True)
