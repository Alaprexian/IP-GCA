"""IP-GCA Copilot — Desktop Application Entrypoint.

Powered by PyWebView with Microsoft Edge WebView2 on Windows.
Connects Python-based IP-GCA Copilot Orchestrator to modern web frontend.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is in sys.path for direct execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("IPGCA.DesktopApp")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="IP-GCA Copilot — Interfaz Gráfica de Escritorio (PyWebView)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Habilita las herramientas de desarrollador Chromium DevTools en la ventana."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1380,
        help="Ancho inicial de la ventana en píxeles (default: 1380)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=880,
        help="Alto inicial de la ventana en píxeles (default: 880)."
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Usa LLM real via Ollama/OpenAI en lugar de respuestas paramétricas simuladas."
    )
    args = parser.parse_args()

    # Verify PyWebView availability
    try:
        import webview
    except ImportError:
        logger.error(
            "PyWebView no está instalado en el entorno actual. "
            "Ejecuta: pip install pywebview pythonnet"
        )
        sys.exit(1)

    # Import the Desktop Bridge API
    from src.ui.app_api import IPGCADesktopAPI

    logger.info("Iniciando orquestador IP-GCA Copilot...")
    logger.info("Cargando base de conocimiento (25 docs) y modelos de inferencia, por favor espere...")
    api = IPGCADesktopAPI(use_real_llm=args.real_llm)
    logger.info("✓ Orquestador y modelos cargados exitosamente.")

    # Locate static HTML entrypoint
    base_dir = Path(__file__).resolve().parent
    html_path = base_dir / "static" / "index.html"

    if not html_path.exists():
        logger.error(f"Archivo HTML no encontrado en: {html_path}")
        sys.exit(1)

    logger.info(f"Cargando interfaz desde: {html_path}")
    logger.info("Creando ventana de PyWebView...")

    # Create Desktop Window
    window = webview.create_window(
        title="IP-GCA Copilot — Sistema de Decisión Asistido por IA (GPL-RAG & HITL)",
        url=str(html_path),
        js_api=api,
        width=args.width,
        height=args.height,
        min_size=(1050, 680),
        background_color="#070B14",
        text_select=True,
    )

    logger.info("Ventana configurada exitosamente. Lanzando bucle de eventos...")
    # Start the event loop (blocking until window is closed)
    webview.start(debug=args.debug)
    logger.info("Aplicación de escritorio finalizada.")


if __name__ == "__main__":
    main()
