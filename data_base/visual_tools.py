"""
Visual Verification Tools for Agentic RAG

Provides tools for AI agents to request re-examination of images
when text context is insufficient for answering specific questions.

Phase 9: Agentic Visual Verification
"""

# Standard library
import logging
import os
from typing import Dict, Any

# Configure logging
logger = logging.getLogger(__name__)

# Security: Allowed image extensions
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif'}

# Base upload folder for path validation
BASE_UPLOAD_FOLDER = "uploads"


def _validate_image_path(image_path: str, user_id: str) -> tuple[bool, str]:
    """
    Validates image path for security.

    Prevents path traversal attacks and ensures path is within user's folder.

    Args:
        image_path: Image path from LLM request.
        user_id: User ID for folder validation.

    Returns:
        Tuple of (is_valid, error_message).
    """
    # Normalize path
    image_path = os.path.normpath(image_path)
    
    # Check for directory traversal attempts
    if ".." in image_path:
        logger.warning(f"Path traversal attempt detected: {image_path}")
        return False, "Invalid path: directory traversal not allowed"
    
    # Build expected user folder prefix
    user_folder = os.path.normpath(os.path.join(BASE_UPLOAD_FOLDER, user_id))
    
    # Ensure path starts with user folder
    if not image_path.startswith(user_folder):
        logger.warning(f"Path security violation: {image_path} not in {user_folder}")
        return False, "Invalid path: outside user folder"
    
    # Check file exists
    if not os.path.exists(image_path):
        return False, "Image file not found"
    
    # Check extension
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Invalid image format: {ext}"
    
    return True, ""


async def verify_image_details(
    image_path: str,
    question: str,
    user_id: str,
    original_summary: str = "",
) -> Dict[str, Any]:
    """
    Tool for agent to re-examine a specific image with a targeted question.

    Validates path security before calling Vision Model.
    This is the main entry point for the visual verification tool.

    Args:
        image_path: Image path from retrieved context metadata.
        question: Specific question about the image (max 200 chars).
        user_id: User ID for path validation.
        original_summary: Original image summary for context.

    Returns:
        Dict with keys:
        - success: bool - Whether verification succeeded
        - result: str - The visual analysis result (if success)
        - error: str - Error message (if failed)
    """
    # 1. Validate and truncate question
    question = question.strip()[:200] if question else ""
    if len(question) < 3:
        return {
            "success": False,
            "error": "Question too short (minimum 3 characters)",
            "result": None,
        }
    
    # 2. Security: Validate image path
    is_valid, error_msg = _validate_image_path(image_path, user_id)
    if not is_valid:
        logger.warning(f"Visual tool path validation failed: {error_msg}")
        return {
            "success": False,
            "error": error_msg,
            "result": None,
        }
    
    # 3. Call Vision Model via ImageSummarizer
    try:
        from multimodal_rag.image_summarizer import summarizer
        
        logger.info(f"Executing visual verification: {os.path.basename(image_path)}")
        result = await summarizer.re_examine_image(
            image_path=image_path,
            specific_question=question,
            original_summary=original_summary,
        )
        
        # Check for error responses from the summarizer
        if result.startswith("Error:"):
            return {
                "success": False,
                "error": result,
                "result": None,
            }
        
        logger.info(f"Visual verification completed: {os.path.basename(image_path)}")
        return {
            "success": True,
            "result": result,
            "error": None,
        }
        
    except FileNotFoundError as e:
        logger.error(f"Visual verification file not found: {e}")
        return {
            "success": False,
            "error": f"Image not found: {str(e)}",
            "result": None,
        }
    except ValueError as e:
        logger.error(f"Visual verification value error: {e}")
        return {
            "success": False,
            "error": f"Invalid input: {str(e)}",
            "result": None,
        }
    except (RuntimeError, OSError) as e:
        logger.error(f"Visual verification failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": f"Vision analysis failed: {str(e)}",
            "result": None,
        }
