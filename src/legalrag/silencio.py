"""Salida limpia: sin barras de descarga, avisos de Hugging Face ni avisos de funciones obsoletas de terceros.

Se llama al comienzo de `cli.main` y de `src/main.py`, antes de cargar modelos. Los errores siguen saliendo.
"""
import logging
import os
import warnings


def silenciar():
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_VERBOSITY", "error")  # el aviso de «unauthenticated requests»
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", message=r".*unauthenticated requests.*")
    for nombre in ("huggingface_hub", "sentence_transformers", "transformers", "accelerate"):
        logging.getLogger(nombre).setLevel(logging.ERROR)
    try:
        from transformers.utils import logging as registro
        registro.set_verbosity_error()
        registro.disable_progress_bar()
    except ImportError:
        pass
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except ImportError:
        pass
