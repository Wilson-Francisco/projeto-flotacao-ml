import logging
import sys

def configurando_logger():
    """Configura o formato padrão de logs do sistema para monitoramento na planta."""
    logger = logging.getLogger("FlotationML")
    
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatador = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Log no terminal
        handler_console = logging.StreamHandler(sys.stdout)
        handler_console.setFormatter(formatador)
        logger.addHandler(handler_console)
        
    return logger
