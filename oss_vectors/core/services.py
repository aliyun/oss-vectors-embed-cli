"""Core services for OSS Vectors operations with user agent tracking."""

import json
import base64
import uuid
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
import alibabacloud_oss_v2 as oss
import alibabacloud_oss_v2.vectors as oss_vectors
from oss_vectors.utils.config import get_user_agent
from oss_vectors.utils.config import get_region, setup_oss_cfg
from oss_vectors.utils.models import get_model_info
import dashscope
from http import HTTPStatus


class DashScopeService:
    """Service for DashScope embedding operations."""
    
    def __init__(self, cfg: oss.Config, region: str, debug: bool = False, console=None):
        # Create DashScope clients with user agent tracking
        self.dash_scope_runtime = dashscope
        # Create OSS client
        region = get_region(region)
        cfg = setup_oss_cfg(region=region)
        cfg.endpoint = f"http://oss-{region}.aliyuncs.com"
        self.oss_client = oss.Client(cfg)
        self.debug = debug
        self.console = console
        
        if self.debug and self.console:
            self.console.print(f"[dim] DashScopeService initialized for region: {region}[/dim]")
            self.console.print(f"[dim] User agent: {get_user_agent()}[/dim]")
    
    def _debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug and self.console:
            self.console.print(f"[dim] {message}[/dim]")
    
    def is_async_model(self, model_id: str) -> bool:
        """Check if model requires async processing."""
        model = get_model_info(model_id)
        return model.is_async() if model else False
    
    def _extract_input_from_payload(self, payload: Dict[str, Any], supports_multimodal_input: bool) -> Any:
        if not supports_multimodal_input:
            return payload.get('inputText')
        else:
            input_list = []
            multimodal_keys = ['text', 'video', 'image', 'multi_images']
            for key in multimodal_keys:
                if key in payload and isinstance(payload[key], dict):
                    input_text = payload[key].get('inputText')
                    if input_text:  # Only add non-empty inputText values
                        input_list.append(input_text)
            if not input_list:
                input_text = payload.get('inputText')
                if input_text:  # Only add non-empty inputText values
                    input_list.append(input_text)
            return input_list

    def embed_with_payload(self, model, payload: Dict[str, Any]) -> List[float]:
        """Embed using direct DashScope API payload for sync models."""
        start_time = time.time()
        model_id = model.model_id
        self._debug_log(f"Starting embedding with model: {model_id}")
        self._debug_log(f"Payload: {json.dumps(payload, indent=2)}")
        
        try:

            modal_type = model.capabilities.modal_type

            supports_multimodal_input = model.capabilities.supports_multimodal_input
            input = self._extract_input_from_payload(payload, supports_multimodal_input)
            
            # Dynamically get the module based on modal_type and call its method
            module = getattr(self.dash_scope_runtime, modal_type)
            response = module.call(
                model=model_id,
                input=input,
                **{k: v for k, v in payload.items() if k not in ['inputText']}
            )

            if response['status_code'] != HTTPStatus.OK:
                error_msg = (f"DashScope embedding failed - Status: {response.get('status_code')}, "
                           f"Request ID: {response.get('request_id', 'N/A')}, "
                           f"Code: {response.get('code', 'N/A')}, "
                           f"Message: {response.get('message', 'N/A')}, "
                           f"Output: {response.get('output', 'N/A')}, "
                           f"Usage: {response.get('usage', 'N/A')}")
                self._debug_log(error_msg)
                raise Exception(error_msg)
            
            elapsed_time = time.time() - start_time
            self._debug_log(f"DashScope API call completed in {elapsed_time:.2f} seconds")
            
            response_body = response['output']
            
            if self.debug and self.console:
                self._debug_log(f"Response request id: {response['request_id']}")
            
            # Extract embedding using schema-based approach
            embedding = model.extract_embedding(response_body)
            
            self._debug_log(f"Generated embedding with {len(embedding)} dimensions")
            total_time = time.time() - start_time
            self._debug_log(f"Total embedding operation completed in {total_time:.2f} seconds")
            
            return embedding

        except Exception as e:
            self._debug_log(f"Unexpected error in embed_with_payload: {str(e)}")
            raise
    
    def _extract_job_id_from_arn(self, invocation_arn: str) -> str:
        """Extract DashScope task ID."""
        # format: https://dashscope.aliyuncs.com/api/v1/tasks/{task_id}
        return invocation_arn.split('/')[-1]

    def embed_async_with_payload(self, model_id: str, final_payload: Dict[str, Any], 
                               batch_text_url: str) -> tuple[List[Dict], str]:
        """Handle async embedding with model_id, final payload, and batch text URL."""
        if not self.is_async_model(model_id):
            raise ValueError(f"Model {model_id} is not an async model")
        
        # For batch text embedding, the inputText should directly correspond to the batch_text_url
        url = batch_text_url  # inputText corresponds directly to the batch_text_url
        
        # Get text_type from final_payload, default to 'document'
        text_type = final_payload.get('text_type', 'document')
        
        # Extract any additional parameters from the payload
        additional_params = {k: v for k, v in final_payload.items() 
                           if k not in ['input', 'inputText', 'url', 'text_type']}
        
        self._debug_log(f"Starting async embedding with model: {model_id}")
        self._debug_log(f"URL: {url}")
        self._debug_log(f"text_type: {text_type}")
        self._debug_log(f"Additional parameters: {additional_params}")

        try:
            # Call the async method
            response = self.dash_scope_runtime.BatchTextEmbedding.async_call(
                model=model_id,
                url=url,
                text_type=text_type,
                **additional_params
            )
            
            # Check response status - expect status_code in the response
            if response.status_code != HTTPStatus.OK:
                error_msg = (f"DashScope async embedding failed - Status: {response.status_code}, "
                           f"Request ID: {getattr(response, 'request_id', 'N/A')}, "
                           f"Code: {getattr(response, 'code', 'N/A')}, "
                           f"Message: {getattr(response, 'message', 'N/A')}, "
                           f"Output: {getattr(response, 'output', 'N/A')}, "
                           f"Usage: {getattr(response, 'usage', 'N/A')}")
                self._debug_log(error_msg)
                raise Exception(error_msg)
            
            # Extract task ID
            task_id = None
            if hasattr(response, 'output') and response.output:
                output = response.output
                if isinstance(output, dict):
                    # Look for task_id in the output
                    task_id = output.get('task_id')
            
            if not task_id:
                raise Exception(f"Could not extract task ID from response: {response}")
            
            self._debug_log(f"Async job started: Task ID: {task_id}")
            
            # Wait for completion and retrieve results
            results = self._wait_and_retrieve_results(task_id)
            
            # Return results with task ID
            return results, task_id
                
        except Exception as e:
            self._debug_log(f"Async embedding failed: {str(e)}")
            raise Exception(f"Async embedding failed: {e}")

    def _wait_and_retrieve_results(self, task_id: str) -> List[Dict]:
        """Wait for DashScope async job completion and retrieve results."""
        self._debug_log(f"Waiting for DashScope async job completion: {task_id}")

        # Use the wait method to wait for task completion
        try:
            # Wait for the task to complete - this will internally poll until completion
            response = self.dash_scope_runtime.BatchTextEmbedding.wait(task_id)
            
            if hasattr(response, 'status_code') and response.status_code != HTTPStatus.OK:
                raise Exception(f"Failed to wait for task completion: {response.status_code}, "
                               f"Code: {getattr(response, 'code', 'N/A')}, "
                               f"Message: {getattr(response, 'message', 'N/A')}")
            
            # Check the final status
            status = 'Unknown'
            if hasattr(response, 'output') and response.output:
                output = response.output
                if isinstance(output, dict):
                    # Check task_status
                    status_info = output.get('task_status', output)
                    if isinstance(status_info, dict):
                        status = status_info.get('task_status', status_info.get('status', 'Unknown'))
                    elif isinstance(status_info, str):
                        status = status_info
            
            # Check if the task completed successfully
            if status not in ['SUCCEEDED', 'Completed', 'COMPLETED']:
                failure_message = getattr(response, 'message', 'Unknown error')
                if hasattr(response, 'output') and response.output and isinstance(response.output, dict):
                    failure_message = response.output.get('message', failure_message)
                raise Exception(f"DashScope async embedding did not complete successfully. Status: {status}, Message: {failure_message}")
            
            self._debug_log(f"Task completed successfully: {task_id}")
            
        except Exception as e:
            self._debug_log(f"Error waiting for task completion: {str(e)}")
            raise Exception(f"Failed to wait for DashScope job completion: {e}")
        
        # Return the response data from the completed task
        # The actual embeddings should be retrieved from the URL provided in the response
        if hasattr(response, 'output') and response.output:
            output = response.output
            if isinstance(output, dict):
                # Extract the results from the response
                return [output]
        
        return []
    
    def _get_results_from_oss(self, output_oss_uri: str) -> List[Dict]:
        """Retrieve results from OSS output location."""
        # Parse OSS URI
        if not output_oss_uri.startswith('oss://'):
            raise ValueError(f"Invalid OSS URI: {output_oss_uri}")

        path_part = output_oss_uri[6:]  # Remove 'oss://'

        if '/' not in path_part:
            raise ValueError(f"Invalid URL format: {output_oss_uri}")
        
        bucket, prefix = path_part.split('/', 1)
        
        # Outputs results to output.json
        result_key = f"{prefix}/output.json" if not prefix.endswith('/') else f"{prefix}output.json"
        
        self._debug_log(f"Reading results from oss://{bucket}/{result_key}")
        
        try:
            obj_response = self.oss_client.get_object(oss.GetObjectRequest(
                bucket=bucket,
                key=result_key,
            ))
            with obj_response.body as body:
                result_data = json.loads(body.read().decode('utf-8'))
            
            # Handle format with 'data' array
            if 'data' in result_data and isinstance(result_data['data'], list):
                return result_data['data']
            elif isinstance(result_data, list):
                return result_data
            else:
                return [result_data]
                
        except Exception as e:
            self._debug_log(f"Error retrieving results from OSS: {str(e)}")
            raise Exception(f"Failed to retrieve results from oss://{bucket}/{result_key}: {e}")
    
    def _has_embeddings(self, data):
        """Check if the data contains embeddings."""
        if isinstance(data, dict):
            # Check for common embedding keys
            embedding_keys = ['embedding', 'embeddings', 'vector', 'vectors']
            if any(key in data for key in embedding_keys):
                return True
            # Check format with 'data' array
            if 'data' in data and isinstance(data['data'], list):
                return any(self._has_embeddings(item) for item in data['data'])
        elif isinstance(data, list):
            # Check if any item in the list has embeddings
            return any(self._has_embeddings(item) for item in data)
        return False


class OSSVectorService:
    """Service for OSS Vector operations."""
    def __init__(self, cfg: oss.Config, region: str, debug: bool = False, console=None):
        self.oss_vectors = oss_vectors.Client(cfg)
        self.debug = debug
        self.console = console
        
        if self.debug and self.console:
            self.console.print(f"[dim] OSSVectorService initialized for region: {cfg.region}[/dim]")
            self.console.print(f"[dim] Using endpoint: {cfg.endpoint}[/dim]")
            self.console.print(f"[dim] User agent: {get_user_agent()}[/dim]")
    
    def _debug_log(self, message: str):
        """Log debug message if debug mode is enabled."""
        if self.debug and self.console:
            self.console.print(f"[dim] {message}[/dim]")
    
    def put_vectors_batch(self, bucket_name: str, index_name: str, 
                         vectors: List[Dict[str, Any]]) -> List[str]:
        """Put multiple vectors into OSS vector index using OSS Vectors batch API."""
        start_time = time.time()
        self._debug_log(f"Starting put_vectors_batch operation")
        self._debug_log(f"Bucket: {bucket_name}, Index: {index_name}")
        self._debug_log(f"Batch size: {len(vectors)} vectors")
        
        try:
            # Use OSS Vectors PutVectors API with multiple vectors
            params = {
                "bucket": bucket_name,
                "indexName": index_name,
                "vectors": vectors
            }
            
            self._debug_log(f"Making OSS Vectors put_vectors batch API call")
            if self.debug and self.console:
                self._debug_log(f"API parameters: {json.dumps({k: v for k, v in params.items() if k != 'vectors'})}")


            response = self.oss_vectors.put_vectors(oss_vectors.models.PutVectorsRequest(
                bucket=bucket_name,
                index_name=index_name,
                vectors=vectors,
            ))
            
            elapsed_time = time.time() - start_time
            self._debug_log(f"OSS Vectors put_vectors batch completed in {elapsed_time:.2f} seconds")
            
            # Extract vector IDs from the batch
            vector_ids = [vector["key"] for vector in vectors]
            self._debug_log(f"Batch stored successfully with {len(vector_ids)} vectors")
            
            return vector_ids

        except Exception as e:
            self._debug_log(f"Unexpected error in put_vectors_batch: {str(e)}")
            raise
    
    def query_vectors(self, bucket_name: str, index_name: str,
                      query_embedding: List[float], top_k: int = 5,
                      filter_expr: Optional[str] = None,
                      return_metadata: bool = True,
                      return_distance: bool = True) -> List[Dict[str, Any]]:
        """Query vectors from OSS vector index using OSS Vectors API."""
        try:
            # Use OSS Vectors QueryVectors API
            params = {
                "bucket": bucket_name,
                "indexName": index_name,
                "queryVector": {
                    "float32": query_embedding  # Query vector also needs float32 format
                },
                "topK": top_k,
                "returnMetadata": return_metadata,
                "returnDistance": return_distance
            }
            
            # Add filter if provided - parse JSON string to object
            if filter_expr:
                import json
                try:
                    # Parse the JSON string into a Python object
                    filter_obj = json.loads(filter_expr)
                    params["filter"] = filter_obj
                    if self.debug:
                        self.console.print(f"[dim] Filter parsed successfully: {filter_obj}[/dim]")
                except json.JSONDecodeError as e:
                    if self.debug:
                        self.console.print(f"[dim] Filter JSON parse error: {e}[/dim]")
                    # If it's not valid JSON, pass as string (for backward compatibility)
                    params["filter"] = filter_expr

            response = self.oss_vectors.query_vectors(oss_vectors.models.QueryVectorsRequest(
                bucket=bucket_name,
                index_name=index_name,
                filter=params.get('filter'),
                query_vector=params.get('queryVector'),
                return_distance=return_distance,
                return_metadata=return_metadata,
                top_k=top_k
            ))
            
            # Process response
            results = []
            if hasattr(response, 'vectors') and response.vectors:
                for vector in response.vectors:
                    result = {
                        'vectorId': vector.get('key'),
                        'similarity': vector.get('distance', 0.0),
                        'metadata': vector.get('metadata', {})
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            raise Exception(f"OSS Vectors query_vectors failed: {e}")
    
    def get_index(self, bucket_name: str, index_name: str) -> Dict[str, Any]:
        """Get index information including dimensions from OSS Vectors API."""
        try:
            # Use OSS Vectors GetIndex API
            response = self.oss_vectors.get_vector_index(oss_vectors.models.GetVectorIndexRequest(
                bucket=bucket_name,
                index_name=index_name,
            ))
            return response.index
            
        except Exception as e:
            raise Exception(f"OSS Vectors get_index failed: {e}")
