def resolve_tag(mode, tag=None, default="baseline_cnn"):
    """Keep synthetic and research artifacts in separate namespaces."""
    if mode not in {"demo", "research"}:
        raise ValueError(f"Unknown mode: {mode}")
    if tag is None:
        return f"demo_{default}" if mode == "demo" else default
    if not tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in tag):
        raise ValueError("Tags may contain letters, numbers, underscores and hyphens only")
    if (mode == "demo") != tag.startswith("demo_"):
        raise ValueError("Demo tags must start with demo_; research tags must not")
    return tag
