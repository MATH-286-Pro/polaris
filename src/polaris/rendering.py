"""Rendering settings shared by evaluation entrypoints and environments."""


def configure_rtx_scene_lighting() -> None:
    """Keep camera lighting independent from GUI/headless launch mode."""

    try:
        import carb
    except ImportError:
        return

    settings = carb.settings.get_settings()
    for key, value in {
        "/rtx/scene/useViewLightingMode": False,
        "/rtx/useViewLightingMode": False,
        "/persistent/rtx/scene/useViewLightingMode": False,
        "/persistent/rtx/useViewLightingMode": False,
        "/rtx/rendermode": "RayTracedLighting",
        "/rtx/reflections/enabled": True,
        "/rtx/reflections/maxBounces": 2,
        "/rtx/raytracing/reflections/enabled": True,
        "/rtx/raytracing/reflections/maxBounces": 2,
        "/rtx/ambientLight/enabled": True,
        "/rtx/ambientLight/intensity": 0.2,
        "/rtx/ambientLight/color": (1.0, 1.0, 1.0),
        "/rtx/sceneDb/ambientLightIntensity": 1.0,
    }.items():
        settings.set(key, value)
