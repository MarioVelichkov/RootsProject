"""Computer-vision pipeline modules.

Heavy TensorFlow imports are loaded lazily so lightweight helpers such as
coordinate conversion can be tested without initializing the model stack.
"""

__all__ = ["RootAnalysisPipeline", "RootPipelineResult"]


def __getattr__(name):
    if name in __all__:
        from .pipeline import RootAnalysisPipeline, RootPipelineResult

        return {
            "RootAnalysisPipeline": RootAnalysisPipeline,
            "RootPipelineResult": RootPipelineResult,
        }[name]
    raise AttributeError(name)
