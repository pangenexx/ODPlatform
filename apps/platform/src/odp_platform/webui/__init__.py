"""Gradio web UI for ODPlatform."""


def create_app():
    from odp_platform.webui.app import create_app as _create_app

    return _create_app()

__all__ = ["create_app"]
