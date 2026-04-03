"""
monitoring.py — LangSmith tracing + OpenTelemetry span support

LangSmith traces every LangGraph node automatically when env vars are set.
This module also exposes a simple @traceable decorator and get_tracer() 
for manual spans in FastAPI routes.
"""

import os
import functools
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

# ---- LangSmith setup (auto-traces LangGraph if env vars present) ----
# Set these in your .env:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=your_key
#   LANGCHAIN_PROJECT=QueryForge

def setup_langsmith():
    langsmith_key = os.getenv("LANGCHAIN_API_KEY")
    if langsmith_key:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
        os.environ.setdefault("LANGCHAIN_PROJECT", "QueryForge")
        print("[monitoring] LangSmith tracing enabled")
    else:
        print("[monitoring] LANGCHAIN_API_KEY not set — LangSmith tracing disabled")


setup_langsmith()


# ---- OpenTelemetry tracer (for FastAPI route spans) ----

_tracer_provider = TracerProvider()
_tracer_provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(_tracer_provider)

def get_tracer():
    return trace.get_tracer("queryforge")


# ---- @traceable decorator ----

def traceable(name: str = None):
    """
    Decorator to add a named span around any function.
    Works with both sync and async functions.
    """
    def decorator(fn):
        span_name = name or fn.__name__

        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            with get_tracer().start_as_current_span(span_name):
                return await fn(*args, **kwargs)

        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            with get_tracer().start_as_current_span(span_name):
                return fn(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(fn):
            return async_wrapper
        return sync_wrapper

    return decorator