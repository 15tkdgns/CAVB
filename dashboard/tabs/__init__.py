"""
Dashboard tabs module
"""
from .tab_overview import render_overview
from .tab_methodology import render_methodology
from .tab_results import render_results
from .tab_validation import render_validation
from .tab_references import render_references


__all__ = [
    'render_overview',
    'render_methodology',
    'render_results',
    'render_validation',
    'render_references',

]
