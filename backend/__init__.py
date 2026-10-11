"""Insurance Policy Explainer - Core Backend Package."""

__version__ = "0.1.0"

# Expose app for convenience
try:
    import frontend.app as app
except ImportError:
    pass
