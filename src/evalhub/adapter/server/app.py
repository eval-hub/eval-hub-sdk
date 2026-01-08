"""Server application for running framework adapters."""

import logging
import signal
import sys
from typing import Any

# typing imports removed - using PEP 604 union syntax
import uvicorn
from fastapi import FastAPI

from ...utils import setup_logging
from ..api.router import AdapterAPIRouter
from ..models.framework import AdapterConfig, FrameworkAdapter

logger = logging.getLogger(__name__)


class AdapterServer:
    """Server for running framework adapters with the standard SDK API."""

    def __init__(self, adapter: FrameworkAdapter):
        """Initialize the server with a framework adapter.

        Args:
            adapter: The framework adapter to run
        """
        self.adapter = adapter
        self.router = AdapterAPIRouter(adapter)
        self.app = self.router.get_app()
        setup_logging(level=self.adapter.config.log_level, stream=sys.stdout)

    def run(
        self,
        host: str | None = None,
        port: int | None = None,
        workers: int | None = None,
        reload: bool = False,
        **kwargs: Any,
    ) -> None:
        """Run the adapter server.

        Args:
            host: Host to bind to (overrides config)
            port: Port to bind to (overrides config)
            workers: Number of workers (overrides config)
            reload: Enable auto-reload for development
            **kwargs: Additional arguments passed to uvicorn.run
        """
        config = self.adapter.config

        # Use provided values or fall back to config
        run_host = host or config.host
        run_port = port or config.port
        run_workers = workers or config.workers

        logger.info(
            f"Starting {config.adapter_name} server on {run_host}:{run_port} "
            f"with {run_workers} worker(s)"
        )

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

        try:
            uvicorn.run(
                self.app,
                host=run_host,
                port=run_port,
                workers=run_workers if not reload else 1,  # Single worker for reload
                reload=reload,
                log_level=config.log_level.lower(),
                access_log=True,
                **kwargs,
            )
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
        except Exception as e:
            logger.exception(f"Server error: {e}")
            sys.exit(1)

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for graceful shutdown."""

        def signal_handler(signum: int, frame: Any) -> None:
            logger.info(f"Received signal {signum}, initiating shutdown...")
            # The adapter shutdown will be handled by FastAPI's shutdown event
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def run_async(
        self, host: str | None = None, port: int | None = None, **kwargs: Any
    ) -> None:
        """Run the server asynchronously.

        Useful for embedding the server in other applications.

        Args:
            host: Host to bind to
            port: Port to bind to
            **kwargs: Additional uvicorn config options
        """
        config = self.adapter.config

        uvicorn_config = uvicorn.Config(
            self.app,
            host=host or config.host,
            port=port or config.port,
            log_level=config.log_level.lower(),
            **kwargs,
        )

        server = uvicorn.Server(uvicorn_config)

        try:
            await server.serve()
        except KeyboardInterrupt:
            logger.info("Server stopped")
        except Exception as e:
            logger.exception(f"Server error: {e}")
            raise


def create_adapter_app(adapter: FrameworkAdapter) -> FastAPI:
    """Create a FastAPI application for the given adapter.

    This function creates a FastAPI app configured with the adapter's API router.
    Useful for testing and embedding the adapter in other applications.

    Args:
        adapter: The framework adapter instance

    Returns:
        FastAPI application instance
    """
    router = AdapterAPIRouter(adapter)
    return router.get_app()


def run_adapter_server(
    adapter_class: type[FrameworkAdapter], config: AdapterConfig, **server_kwargs: Any
) -> None:
    """Convenience function to create and run an adapter server.

    Args:
        adapter_class: The FrameworkAdapter class to instantiate
        config: Configuration for the adapter
        **server_kwargs: Additional arguments for the server
    """
    # Create adapter instance
    adapter = adapter_class(config)

    # Create and run server
    server = AdapterServer(adapter)
    server.run(**server_kwargs)
