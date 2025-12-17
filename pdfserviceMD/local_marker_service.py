"""
Local Marker PDF OCR Service

Provides local PDF-to-Markdown conversion using the Marker library.
This is an alternative to the Datalab API that runs entirely on-device.

Marker is developed by VikParuchuri (USA) - GPL License
https://github.com/VikParuchuri/marker
"""

# Standard library
import logging
import os
from typing import Tuple, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Lazy-loaded marker components
_marker_converter = None


class LocalMarkerError(Exception):
    """Raised when local Marker processing fails."""
    pass


def _get_marker_converter():
    """
    Lazily initializes and returns the Marker converter.
    
    Returns:
        Initialized Marker converter instance.
    """
    global _marker_converter
    
    if _marker_converter is None:
        logger.info("Initializing local Marker converter...")
        try:
            # Check if GPU should be used
            marker_gpu_env = os.getenv("MARKER_USE_GPU", "false")
            use_gpu = marker_gpu_env.lower() in ("true", "1", "yes")
            logger.info(f"MARKER_USE_GPU env: '{marker_gpu_env}', use_gpu: {use_gpu}")
            
            # Set CUDA device BEFORE importing torch
            if use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            
            import torch
            
            device = "cpu"  # Default
            if use_gpu:
                try:
                    # Try to actually use CUDA - most reliable detection method
                    test_tensor = torch.tensor([1.0], device="cuda")
                    gpu_name = torch.cuda.get_device_name(0)
                    device = "cuda"
                    del test_tensor
                    torch.cuda.empty_cache()
                    logger.info(f"Using GPU: {gpu_name}")
                except Exception as e:
                    logger.warning(f"GPU not usable, falling back to CPU: {e}")
                    device = "cpu"
            
            if device == "cpu":
                # Force CPU mode for Surya/Marker via environment variables
                os.environ["TORCH_DEVICE"] = "cpu"
                os.environ["RECOGNITION_DEVICE"] = "cpu"
                os.environ["DETECTOR_DEVICE"] = "cpu"
                os.environ["TEXIFY_DEVICE"] = "cpu"
                os.environ["LAYOUT_DEVICE"] = "cpu"
                os.environ["TABLE_REC_DEVICE"] = "cpu"
                logger.info("Using CPU (set MARKER_USE_GPU=true to enable GPU)")
            
            from marker.converters.pdf import PdfConverter
            from marker.models import create_model_dict
            from marker.config.parser import ConfigParser
            
            # Create config with device setting
            config_parser = ConfigParser({"device": device})
            
            # Create model dictionary (this loads the ML models)
            logger.info("Loading Marker models (this may take a moment)...")
            model_dict = create_model_dict()
            
            # Create converter
            _marker_converter = PdfConverter(
                config=config_parser.generate_config_dict(),
                artifact_dict=model_dict,
            )
            
            logger.info(f"Local Marker converter ready (device: {device})")
            
        except ImportError as e:
            logger.error(f"Marker not installed: {e}")
            raise LocalMarkerError(f"Marker library not available: {e}")
        except Exception as e:
            logger.error(f"Failed to initialize Marker: {e}", exc_info=True)
            raise LocalMarkerError(f"Marker initialization failed: {e}")
    
    return _marker_converter


def convert_pdf_to_markdown_local(pdf_path: str) -> Tuple[str, int]:
    """
    Converts a PDF to Markdown using local Marker.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Tuple of (markdown_text, page_count).
        
    Raises:
        LocalMarkerError: If conversion fails.
    """
    if not os.path.exists(pdf_path):
        raise LocalMarkerError(f"PDF file not found: {pdf_path}")
    
    logger.info(f"Starting local Marker conversion: {pdf_path}")
    
    try:
        converter = _get_marker_converter()
        
        # Convert PDF
        result = converter(pdf_path)
        
        # Extract markdown and metadata
        markdown = result.markdown
        
        # Try to get page count from result structure
        page_count = 1
        if hasattr(result, 'pages') and result.pages:
            page_count = len(result.pages)
        elif hasattr(result, 'document') and hasattr(result.document, 'pages'):
            page_count = len(result.document.pages)
        else:
            # Count page breaks in markdown as fallback
            page_break_count = markdown.count('\n---\n')
            if page_break_count > 0:
                page_count = page_break_count + 1
        
        logger.info(f"Local Marker completed: {page_count} pages, {len(markdown)} chars")
        logger.debug(f"Result attributes: {dir(result)}")
        
        return markdown, page_count
        
    except LocalMarkerError:
        raise
    except Exception as e:
        logger.error(f"Local Marker conversion failed: {e}", exc_info=True)
        raise LocalMarkerError(f"PDF conversion failed: {e}")


def build_markdown_with_page_markers_local(markdown: str, page_count: int) -> str:
    """
    Adds [[PAGE_N]] markers to markdown from local Marker.
    
    Marker uses horizontal rules (---) as page separators.
    
    Args:
        markdown: Raw markdown from Marker.
        page_count: Number of pages in the PDF.
        
    Returns:
        Markdown with [[PAGE_N]] markers.
    """
    if not markdown:
        return ""
    
    # Marker uses --- as page separator
    separator_patterns = ["\n---\n\n", "\n\n---\n\n", "\n---\n"]
    
    for sep in separator_patterns:
        if sep in markdown:
            pages = markdown.split(sep)
            if len(pages) >= page_count - 1:  # Allow some tolerance
                markdown_parts = []
                for i, page_md in enumerate(pages):
                    page_md = page_md.strip()
                    if page_md:
                        markdown_parts.append(f"[[PAGE_{i+1}]]\n{page_md}")
                
                if markdown_parts:
                    logger.debug(f"Split into {len(markdown_parts)} pages")
                    return "\n\n".join(markdown_parts)
                break
    
    # No separator found - return with PAGE_1 marker
    logger.debug("No page separator found, treating as single page")
    return f"[[PAGE_1]]\n{markdown}"


def ocr_with_local_marker(pdf_path: str) -> str:
    """
    Main entry point for local Marker OCR.
    
    This is a synchronous function designed to be called via run_in_threadpool.
    
    Args:
        pdf_path: Path to the PDF file.
        
    Returns:
        Markdown text with [[PAGE_N]] markers.
        
    Raises:
        LocalMarkerError: If processing fails.
    """
    markdown, page_count = convert_pdf_to_markdown_local(pdf_path)
    return build_markdown_with_page_markers_local(markdown, page_count)


def is_marker_available() -> bool:
    """
    Checks if the Marker library is installed and importable.
    
    Returns:
        True if Marker is available, False otherwise.
    """
    try:
        import marker
        return True
    except ImportError:
        return False
