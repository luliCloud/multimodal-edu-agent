try:
    from celery import Celery
except ImportError:  # Celery is optional for the Mac-only skeleton.
    Celery = None


def create_celery_app():
    if Celery is None:
        raise RuntimeError("Install the worker extra to enable Celery: pip install -e '.[worker]'")

    return Celery(
        "multimodal_edu_agent",
        broker="redis://localhost:6379/0",
        backend="redis://localhost:6379/1",
    )

