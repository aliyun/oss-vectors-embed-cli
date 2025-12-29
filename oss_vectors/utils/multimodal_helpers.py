"""Multimodal helper utilities for async models handling multi_images, video, text, image content."""

import base64
from typing import Dict, Any


def is_local_path(file_path: str) -> bool:
    """
    Determine if a file path is local or OSS URI.
    
    Args:
        file_path: File path to check
        
    Returns:
        True if local path, False if OSS URI
    """
    return not file_path.startswith('oss://')


def encode_file_to_base64(file_path: str, max_file_size: int = None) -> str:
    """
    Read local file and encode to base64 string.
    
    Args:
        file_path: Local file path
        max_file_size: Maximum file size in bytes (None = no limit)
        
    Returns:
        Base64 encoded string
        
    Raises:
        Exception: If file cannot be read or exceeds size limit
    """
    try:
        with open(file_path, 'rb') as file:
            file_content = file.read()
            
        # Check file size limit if specified
        if max_file_size and len(file_content) > max_file_size:
            raise Exception(f"File size exceeds {max_file_size // (1024*1024)}MB limit: {len(file_content)} bytes")
            
        return base64.b64encode(file_content).decode('utf-8')
    except Exception as e:
        raise Exception(f"Failed to encode file {file_path}: {str(e)}")


def build_media_source(file_path: str, src_bucket_owner: str = None, max_file_size: int = None) -> Dict[str, Any]:
    """
    Build mediaSource object for multimodal models based on file path type.
    
    Args:
        file_path: File path (local or OSS URI)
        src_bucket_owner: Alibaba Cloud account ID for OSS files (optional)
        max_file_size: Maximum local file size in bytes (None = no limit)
        
    Returns:
        Dictionary with either ossLocation or base64String format
    """
    if is_local_path(file_path):
        # Local file - use base64 encoding
        base64_content = encode_file_to_base64(file_path, max_file_size)
        return {"base64String": base64_content}
    else:
        # OSS URI - use ossLocation format
        media_source = {"ossLocation": {"uri": file_path}}
        if src_bucket_owner:
            media_source["ossLocation"]["bucketOwner"] = src_bucket_owner
        return media_source


def create_multimodal_metadata(input_type: str, source_location: str, 
                              embedding_data: Dict, clip_index: int = 0) -> Dict[str, Any]:
    """
    Create metadata for multimodal vector storage (async models handling multi_images, video, text, image).
    
    Args:
        input_type: The input type (text, video, multi_images, image)
        source_location: Original file location
        embedding_data: Embedding response data
        clip_index: Index of the clip (for multi-clip responses)
        
    Returns:
        Metadata dictionary for multimodal content
    """
    metadata = {
        'OSS-VECTORS-EMBED-SRC-LOCATION': source_location
    }
    # Add temporal information if available
    if embedding_data.get('startSec') is not None:
        metadata['OSS-VECTORS-EMBED-START-SEC'] = embedding_data['startSec']
    if embedding_data.get('endSec') is not None:
        metadata['OSS-VECTORS-EMBED-END-SEC'] = embedding_data['endSec']
    
    # Add embedding type if available
    if embedding_data.get('embeddingOption'):
        metadata['OSS-VECTORS-EMBED-TYPE'] = embedding_data['embeddingOption']
    
    return metadata
