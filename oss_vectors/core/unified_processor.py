"""Unified processing pipeline for sync and async models."""

import base64
import gzip
import json
import urllib.request
from io import BytesIO
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import alibabacloud_oss_v2 as oss
from oss_vectors.utils.models import get_model_info, SupportedModel, ProcessingInput, generate_vector_key
from oss_vectors.utils.multimodal_helpers import create_multimodal_metadata


@dataclass
class ProcessingResult:
    """Unified result structure."""
    vectors: List[Dict[str, Any]]  # List of vectors to store
    result_type: str  # "single" or "multiclip"
    job_id: Optional[str] = None
    raw_results: Optional[List[Dict[str, Any]]] = None  # Raw results for timing extraction


class UnifiedProcessor:
    """Unified processor that handles both sync and async models."""
    
    def __init__(self, dash_scope_service, oss_vector_service, cfg=None):
        self.dash_scope_service = dash_scope_service
        self.oss_vector_service = oss_vector_service
        self.cfg = cfg
    
    def process(self, model: SupportedModel, processing_input: ProcessingInput,
                user_dash_scope_params: Dict[str, Any] = None,
                batch_text_url: str = None,  # Replaces async_output_oss_uri
                vector_bucket_name: str = None, index_name: str = None,
                precomputed_dimensions: int = None,
                presign_url=None) -> ProcessingResult:
        """Unified processing method for all input types and models."""
        
        # Step 1: Get index dimensions if available
        # Use pre-computed dimensions (required parameter)
        if precomputed_dimensions is None:
            raise ValueError("Unexpected error occurred. Index dimensions are not fetched.")
        
        index_dimensions = precomputed_dimensions
        
        # Step 2: Build content for schema application
        content = self._prepare_content(processing_input, index_dimensions, user_dash_scope_params, presign_url)
        
        # Step 3: Build payload using schema-based system (includes validation and merge)
        user_dash_scope_params = user_dash_scope_params or {}
        
        # Build final payload with user parameters
        if model.is_async():
            async_config = {
                "batch_text_url": batch_text_url,
                "text_type": user_dash_scope_params.get("text_type", "document")
            }
            final_payload = model.build_payload(processing_input.content_type, content, user_dash_scope_params, async_config)
        else:
            final_payload = model.build_payload(processing_input.content_type, content, user_dash_scope_params)
        
        # Step 4: Get embeddings
        if model.is_async():
            raw_results, job_id = self.dash_scope_service.embed_async_with_payload(
                model.model_id, final_payload, batch_text_url
            )
        else:
            raw_results = self._embed_sync(model.model_id, final_payload, user_dash_scope_params)
            job_id = None
        
        # Step 6: Process results into vectors (unified)
        vectors = self._prepare_vectors(raw_results, processing_input, model)
        
        # Step 7: Determine result type
        result_type = "multiclip" if len(vectors) > 1 else "single"
        
        return ProcessingResult(vectors=vectors, result_type=result_type, job_id=job_id, raw_results=raw_results)
    
    def _prepare_content(self, processing_input: ProcessingInput, index_dimensions: int, user_dash_scope_params: Dict[str, Any] = None, presign_url=None) -> Dict[str, Any]:
        """Prepare content dictionary for schema application."""
        content = {"index": {"dimensions": index_dimensions}}


        if processing_input.content_type == "text":
            content.update(self._prepare_text_content(processing_input))
        elif processing_input.content_type == "batch_text":
            if "batch_text_url" in processing_input.data:
                content["batch_text_url"] = processing_input.data["batch_text_url"]
                content["text_type"] = (user_dash_scope_params or {}).get("text_type", "document")

        elif processing_input.content_type == "image":
            content.update(self._prepare_image_content(processing_input, presign_url))
                
        elif processing_input.content_type == "multimodal":
            # Handle multimodal input (text + image)
            multimodal_data = processing_input.data.get("multimodal", {})
            content["text"] = multimodal_data.get("text", "")
            content["video"] = multimodal_data.get("video", "")
            content["multi_images"] = multimodal_data.get("multi_images", "")
            image_path = multimodal_data.get("image", "")
            if image_path:
                if image_path.startswith('http://') or image_path.startswith('https://'):
                    content["image"] = multimodal_data.get("image", "")
                else:
                    base64_image = self._read_image_as_base64(image_path)

                    # Determine MIME type
                    if image_path.lower().endswith(('.jpg', '.jpeg')):
                        mime_type = "image/jpeg"
                    elif image_path.lower().endswith('.png'):
                        mime_type = "image/png"
                    else:
                        mime_type = "image/jpeg"  # default

                    content["image_base64"] = base64_image
                    content["image"] = f"data:{mime_type};base64,{base64_image}"


        elif processing_input.content_type == "video":
            content.update(self._prepare_video_content(processing_input, presign_url))

        elif processing_input.content_type == "multi_images":
            content.update(self._prepare_multi_images_content(processing_input, presign_url))
        return content
    
    def _read_file_content(self, file_path: str) -> str:
        """Read text file content from local, OSS, or URL."""
        if file_path.startswith('oss://'):
            parts = file_path[6:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''

            response = self.dash_scope_service.oss_client.get_object(oss.GetObjectRequest(
                bucket=bucket,
                key=key,
            ))
            with response.body as body:
                return body.read().decode('utf-8')
        elif file_path.startswith('http://') or file_path.startswith('https://'):
            # Handle URL
            response = urllib.request.urlopen(file_path)
            return response.read().decode('utf-8')
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
    
    def _read_image_as_base64(self, file_path: str) -> str:
        """Read image file and convert to base64."""
        if file_path.startswith('oss://'):
            parts = file_path[6:].split('/', 1)
            bucket = parts[0]
            key = parts[1] if len(parts) > 1 else ''
            response = self.dash_scope_service.oss_client.get_object(oss.GetObjectRequest(
                bucket=bucket,
                key=key,
            ))
            with response.body as body:
                image_bytes = body.read()
        else:
            with open(file_path, 'rb') as f:
                image_bytes = f.read()
        
        return base64.b64encode(image_bytes).decode('utf-8')
    
    def _apply_schema(self, schema: Any, context: dict) -> Any:
        """Recursively apply context to schema template."""
        if isinstance(schema, dict):
            result = {}
            for key, value in schema.items():
                applied_value = self._apply_schema(value, context)
                if applied_value is not None:  # Skip None values
                    result[key] = applied_value
            return result
        elif isinstance(schema, list):
            return [self._apply_schema(item, context) for item in schema]
        elif isinstance(schema, str) and schema.startswith("{") and schema.endswith("}"):
            # Template substitution
            path = schema[1:-1]  # Remove { }
            return self._get_by_path(context, path)
        else:
            return schema
    
    def _get_by_path(self, obj: dict, path: str) -> Any:
        """Get value from nested dict by dot notation path."""
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None  # Skip optional parameters
        return current
    
    def _embed_sync(self, model_id: str, embedding_input: Dict[str, Any],
                    user_dash_scope_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Handle sync embedding using model's schema."""
        from oss_vectors.utils.models import get_model_info
        
        # Get model object for schema-based embedding extraction
        model = get_model_info(model_id)
        if not model:
            raise ValueError(f"Unsupported model: {model_id}")
        
        # The embedding_input is already the correct payload from _prepare_embedding_input
        embedding = self.dash_scope_service.embed_with_payload(model, embedding_input)
        
        return [{"embedding": embedding}]
    
    def _prepare_vectors(self, raw_results: List[Dict[str, Any]], 
                        processing_input: ProcessingInput, model: SupportedModel) -> List[Dict[str, Any]]:
        """Convert raw embedding results to vector storage format."""
        vectors = []
        
        # Check if this is batch text processing with URL results
        if "batch_text_url" in processing_input.data and raw_results and 'url' in raw_results[0]:
            # Handle async batch text processing - download and decompress embeddings from URL
            for i, result in enumerate(raw_results):
                url = result.get('url')
                if not url:
                    raise ValueError(f"Missing URL in batch text result {i+1}")
                
                # Download and decompress the GZ file containing embeddings
                try:
                    response = urllib.request.urlopen(url)
                    compressed_data = response.read()
                    decompressed_data = gzip.decompress(compressed_data)
                    embedding_content = decompressed_data.decode('utf-8')
                    
                    # The embedding content may be in one of two formats:
                    # 1. Single JSON object: {"output":{"code":200,"embedding":[...],"message":"Success",...}}
                    # 2. Multiple JSON lines: Each line is a JSON object with the same structure
                    try:
                        # First, try parsing as single JSON object
                        parsed_data = json.loads(embedding_content)
                        
                        # If successful, handle as single JSON object
                        if isinstance(parsed_data, dict) and "output" in parsed_data and "embedding" in parsed_data["output"]:
                            # Process the single embedding
                            embedding_list = parsed_data["output"]["embedding"]
                            if isinstance(embedding_list, list):
                                embedding = embedding_list
                                # Create vector for this embedding
                                # Generate vector key based on processing input preferences
                                vector_key = generate_vector_key(None, False, processing_input.source_location, processing_input.key_prefix)
                                
                                # Prepare metadata
                                vector_metadata = processing_input.metadata.copy()
                                
                                # Add standard metadata fields for batch text processing
                                content_type_formatted = processing_input.content_type.upper().replace('_', '-')

                                # Get the text index from the response
                                text_index = parsed_data["output"]["text_index"]
                                if text_index is not None:
                                    # Read the source text file to get the corresponding line
                                    source_text = self._read_file_content(processing_input.source_location)
                                    source_lines = source_text.split('\n')

                                vector_metadata["OSS-VECTORS-EMBED-SRC-CONTENT"] = source_lines[text_index].strip()
                                vector_metadata["OSS-VECTORS-EMBED-SRC-INDEX"] = str(text_index)
                                vector_metadata["OSS-VECTORS-EMBED-SRC-CONTENT-TYPE"] = content_type_formatted
                                vector_metadata["OSS-VECTORS-EMBED-SRC-LOCATION"] = processing_input.source_location

                                # Create vector in OSS Vectors API format
                                vector = {
                                    "key": vector_key,
                                    "data": {
                                        "float32": embedding
                                    },
                                    "metadata": vector_metadata
                                }
                                
                                vectors.append(vector)
                            else:
                                raise ValueError(f"Expected embedding to be a list, got {type(embedding_list)}")
                        else:
                            raise ValueError(f"Expected JSON with output.embedding structure, got: {type(parsed_data)}")
                    except json.JSONDecodeError:
                        # If single JSON parsing fails, try parsing as multiple JSON lines
                        lines = embedding_content.strip().split('\n')
                        # Read the source text file to get the corresponding lines
                        source_text = self._read_file_content(processing_input.source_location)
                        source_lines = source_text.split('\n')
                        
                        for line_num, line in enumerate(lines):
                            line = line.strip()
                            if not line:
                                continue  # Skip empty lines
                            
                            try:
                                parsed_line = json.loads(line)
                                
                                # Check if the line has the expected structure: {"output":{"code":200,"embedding":[...]}}
                                if isinstance(parsed_line, dict) and "output" in parsed_line and "embedding" in parsed_line["output"]:
                                    embedding_list = parsed_line["output"]["embedding"]
                                    if isinstance(embedding_list, list):
                                        embedding = embedding_list
                                        # Get the text index from the response
                                        text_index = parsed_line["output"]["text_index"]
                                        
                                        # Generate vector key based on processing input preferences
                                        vector_key = generate_vector_key(None, False, processing_input.source_location, processing_input.key_prefix)
                                        
                                        # Prepare metadata
                                        vector_metadata = processing_input.metadata.copy()
                                        
                                        # Add standard metadata fields for batch text processing
                                        content_type_formatted = processing_input.content_type.upper().replace('_', '-')

                                        vector_metadata["OSS-VECTORS-EMBED-SRC-CONTENT"] = source_lines[text_index].strip()
                                        vector_metadata["OSS-VECTORS-EMBED-SRC-INDEX"] = str(text_index)
                                        vector_metadata["OSS-VECTORS-EMBED-SRC-CONTENT-TYPE"] = content_type_formatted
                                        vector_metadata["OSS-VECTORS-EMBED-SRC-LOCATION"] = processing_input.source_location

                                        
                                        # Create vector in OSS Vectors API format
                                        vector = {
                                            "key": vector_key,
                                            "data": {
                                                "float32": embedding
                                            },
                                            "metadata": vector_metadata
                                        }
                                        
                                        vectors.append(vector)
                                    else:
                                        raise ValueError(f"Expected embedding to be a list in line {line_num+1}, got {type(embedding_list)}")
                                else:
                                    raise ValueError(f"Expected JSON line with output.embedding structure in line {line_num+1}, got: {type(parsed_line)}")
                            except json.JSONDecodeError as e:
                                raise ValueError(f"Failed to parse JSON from line {line_num+1}: {str(e)}")
                    
                    # Ensure embedding is a list of numbers (for single JSON case)
                    if 'embedding' in locals() and embedding:
                        if not isinstance(embedding, (list, tuple)):
                            raise ValueError(f"Embedding must be a list or array of numbers, got {type(embedding)}")
                        
                        # Verify that all elements in embedding are numbers
                        if not all(isinstance(x, (int, float)) for x in embedding):
                            raise ValueError(f"All embedding elements must be numbers, got {[type(x) for x in embedding if not isinstance(x, (int, float))][:5]}")
                    
                except Exception as e:
                    raise ValueError(f"Failed to download or decompress embeddings from URL {url}: {str(e)}")
        else:
            # Handle regular embedding processing (non-batch text)
            for i, result in enumerate(raw_results):
                embedding = result.get('embedding', [])
                if not embedding:
                    raise ValueError(f"Missing required embedding in result {i+1}/{len(raw_results)}. Result: {result}")
                
                # Generate vector key based on processing input preferences
                if processing_input.custom_key and len(raw_results) == 1:
                    # Use custom key only for single vector results
                    vector_key = generate_vector_key(processing_input.custom_key, False, processing_input.source_location, processing_input.key_prefix)
                elif processing_input.filename_as_key and len(raw_results) == 1:
                    # Use object key/filename only for single vector results
                    vector_key = generate_vector_key(None, True, processing_input.source_location, processing_input.key_prefix)
                else:
                    # Generate UUID for multi-vector results or when no key preference specified
                    vector_key = generate_vector_key(None, False, processing_input.source_location, processing_input.key_prefix)
                
                # Prepare metadata
                vector_metadata = processing_input.metadata.copy()
                
                # Add standard metadata fields based on input type
                if "file_path" in processing_input.data:
                    # File input (--text, --image, --video, --multi_images) - always add location
                    content_type_formatted = processing_input.content_type.upper().replace('_', '-')
                    vector_metadata["OSS-VECTORS-EMBED-SRC-LOCATION"] = processing_input.source_location
                    vector_metadata["OSS-VECTORS-EMBED-SRC-CONTENT-TYPE"] = content_type_formatted
                    
                    # For text files, also add the raw text content
                    if processing_input.content_type == "text" and "text" in processing_input.data:
                        vector_metadata["OSS-VECTORS-EMBED-SRC-CONTENT"] = processing_input.data["text"]
                    # For image/video/multi_images files, OSS-VECTORS-EMBED-SRC-CONTENT is not added (blank)
                elif processing_input.content_type == "multimodal":
                    # Multimodal input - add both content and location
                    multimodal_data = processing_input.data.get("multimodal", {})
                    content_type_formatted = processing_input.content_type.upper()
                    vector_metadata["OSS-VECTORS-EMBED-SRC-CONTENT-TYPE"] = content_type_formatted
                    if multimodal_data.get("text", ""):
                        vector_metadata["OSS-VECTORS-EMBED-SRC-CONTENT"] = multimodal_data.get("text", "")
                    if multimodal_data.get("image", ""):
                        vector_metadata["OSS-VECTORS-EMBED-SRC-LOCATION"] = multimodal_data.get("image", "")
                    if multimodal_data.get("video", ""):
                        vector_metadata["OSS-VECTORS-EMBED-SRC-LOCATION"] = multimodal_data.get("video", "")
                    if multimodal_data.get("multi_images", ""):
                        vector_metadata["OSS-VECTORS-EMBED-SRC-LOCATION"] = multimodal_data.get("multi_images", "")
                else:
                    # Direct text input (--text-value) - only add content, no location
                    if processing_input.content_type == "text" and "text" in processing_input.data:
                        vector_metadata["OSS-VECTORS-EMBED-SRC-CONTENT"] = processing_input.data["text"]
                    content_type_formatted = processing_input.content_type.upper()
                    vector_metadata["OSS-VECTORS-EMBED-SRC-CONTENT-TYPE"] = content_type_formatted
                    vector_metadata["OSS-VECTORS-EMBED-SRC-LOCATION"] = processing_input.source_location
                
                # Add model-specific metadata for async models
                if model.is_async():
                    vector_metadata.update(create_multimodal_metadata(
                        processing_input.content_type, processing_input.source_location, result, i
                    ))
                
                # Create vector in OSS Vectors API format
                vector = {
                    "key": vector_key,
                    "data": {
                        "float32": embedding
                    },
                    "metadata": vector_metadata
                }
                
                vectors.append(vector)
        
        return vectors
    
    def store_vectors(self, vectors: List[Dict[str, Any]], vector_bucket_name: str, 
                     index_name: str) -> List[str]:
        """Store vectors using batch operation."""
        if not vectors:
            return []
        
        self.oss_vector_service.put_vectors_batch(vector_bucket_name, index_name, vectors)
        return [v["key"] for v in vectors]

    def _prepare_text_content(self, processing_input: ProcessingInput) -> Dict[str, Any]:
        """Prepare content for text processing input."""
        content = {}
        if "file_path" in processing_input.data and "text" not in processing_input.data:
            # Read file content for sync models
            file_content = self._read_file_content(processing_input.data["file_path"])
            content["text"] = file_content
            # Update processing_input.data so metadata logic can access the text content
            processing_input.data["text"] = file_content
        else:
            content["text"] = processing_input.data.get("text", "")
        return content

    def _prepare_image_content(self, processing_input: ProcessingInput, presign_url=None) -> Dict[str, Any]:
        """Prepare content for image processing input."""
        content = {}
        if "file_path" in processing_input.data:
            file_path = processing_input.data["file_path"]
            if file_path.startswith('http://') or file_path.startswith('https://'):
                content["image"] = processing_input.data.get("file_path", "")
            elif presign_url and file_path.startswith('oss://'):
                pre_result = self._generate_oss_presign_url(file_path)
                content["image"] = f"{pre_result.url}"
            else:
                # For async models , preserve file_path for media_source
                content["file_path"] = file_path

                # For sync models, read and encode image
                base64_image = self._read_image_as_base64(file_path)

                # Set both formats to support different models:
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    mime_type = "image/jpeg"
                elif file_path.lower().endswith('.png'):
                    mime_type = "image/png"
                else:
                    mime_type = "image/jpeg"  # default

                content["image_base64"] = base64_image
                content["image"] = f"data:{mime_type};base64,{base64_image}"
        else:
            content["image_base64"] = processing_input.data.get("image_base64", "")
            content["image"] = processing_input.data.get("image", "")
        return content

    def _prepare_video_content(self, processing_input: ProcessingInput, presign_url=None) -> Dict[str, Any]:
        """Prepare content for video processing input."""
        content = {}
        file_path = processing_input.data.get("file_path", "")
        if presign_url and file_path.startswith('oss://'):
            pre_result = self._generate_oss_presign_url(file_path)
            content["video"] = f"{pre_result.url}"
        else:
            content["video"] = processing_input.data.get("file_path", "")
        return content

    def _prepare_multi_images_content(self, processing_input: ProcessingInput, presign_url=None) -> Dict[str, Any]:
        """Prepare content for multi_images processing input."""
        content = {}
        if "file_path" in processing_input.data:
            file_path = processing_input.data["file_path"]
            validated_paths = []
            for path in file_path:
                if isinstance(path, str) and (path.startswith('http://') or path.startswith('https://')):
                    validated_paths.append(path)
                elif presign_url and path.startswith('oss://'):
                    pre_result = self._generate_oss_presign_url(path)
                    validated_paths.append(pre_result)
                elif isinstance(path, str):
                    base64_image = self._read_image_as_base64(path)

                    # Determine MIME type
                    if path.lower().endswith(('.jpg', '.jpeg')):
                        mime_type = "image/jpeg"
                    elif path.lower().endswith('.png'):
                        mime_type = "image/png"
                    else:
                        mime_type = "image/jpeg"  # default

                    # Format as data URI
                    validated_paths.append(f"data:{mime_type};base64,{base64_image}")
            content["multi_images"] = validated_paths
        else:
            content["multi_images"] = processing_input.data.get("multi_images", "")
        return content

    def _generate_oss_presign_url(self, oss_path: str) -> str:
        """Generate presign URL for OSS path."""
        parts = oss_path[6:].split('/', 1)
        bucket = parts[0]
        key = parts[1] if len(parts) > 1 else ''
        pre_result = self.dash_scope_service.oss_client.presign(oss.GetObjectRequest(
            bucket=bucket,
            key=key,
        ))
        return pre_result

