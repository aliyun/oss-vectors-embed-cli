import os
import json
import click
from rich.progress import Progress, SpinnerColumn, TextColumn

from oss_vectors.core.services import DashScopeService, OSSVectorService
from oss_vectors.core.unified_processor import UnifiedProcessor
from oss_vectors.utils.config import get_region
from oss_vectors.utils.models import get_model_info, validate_user_parameters, prepare_processing_input, determine_content_type
from oss_vectors.core.streaming_batch_orchestrator import StreamingBatchOrchestrator


def _create_progress_context(console):
    """Create progress context for operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    )


def _validate_inputs(text_value, text, image, video, multi_images, model, key, filename_as_key, batch_text_url):
    """Validate input parameters."""
    inputs_provided = sum(bool(x) for x in [text_value, text, image, video, multi_images, batch_text_url])

    if inputs_provided == 0:
        raise click.ClickException("At least one input must be provided: --text-value, --text, --image, --video, or --multi_images")

    # Check mutual exclusivity of key parameters
    if key and filename_as_key:
        raise click.ClickException("Cannot use both --key and --filename-as-key. Choose one.")

    # --filename-as-key not allowed with --text-value (no file/object to extract name from)
    if filename_as_key and text_value:
        raise click.ClickException("--filename-as-key is not supported with --text-value (no file or object to extract name from)")

    # Key parameters not supported for video/multi_images (multi-vector output)
    if (key or filename_as_key) and (video or multi_images):
        raise click.ClickException("--key and --filename-as-key are not supported for video/multi_images inputs (multi-vector output)")

    # Special case: Allow multimodal input for supported models
    is_multimodal_input = model.supports_multimodal_input() and sum(
        1 for inp in [text_value, text, image, video, multi_images] if inp is not None
    ) >= 2

    if inputs_provided > 1 and not is_multimodal_input:
        raise click.ClickException("Only one input type can be specified at a time, except for multimodal input with supported models")

    return is_multimodal_input


@click.command()
@click.option('--vector-bucket-name', required=True, help='OSS vector bucket name')
@click.option('--index-name', required=True, help='Vector index name')
@click.option('--model-id', required=True, help='DashScope embedding model ID')
@click.option('--text-value', help='Direct text input to embed')
@click.option('--text', help='Text file path (local file or OSS URI)')
@click.option('--image', help='Image file path (local file or OSS URI)')
@click.option('--video', help='Video file path (local file)')
@click.option('--dashscope-inference-params', help='DashScope inference parameters (JSON)')
@click.option('--key', help='Custom vector key (auto-generated UUID if not provided)')
@click.option('--key-prefix', help='Prefix to prepend to all vector keys')
@click.option('--filename-as-key', is_flag=True, help='Use filename as vector key')
@click.option('--metadata', help='Additional metadata (JSON)')
@click.option('--max-workers', type=int, default=4, help='Maximum parallel workers for batch processing')
@click.option('--batch-size', type=click.IntRange(1, 500), default=500, help='Number of vectors per OSS Vector put_vectors call (1-500, default: 500)')
@click.option('--output', type=click.Choice(['json', 'table']), default='json', help='Output format')
@click.option('--region', help='OSS region name (effective in OSS path mode)')
@click.pass_context
def embed_put(ctx, vector_bucket_name, index_name, model_id, text_value, text, image,
              video, dashscope_inference_params, key, key_prefix, filename_as_key,
              metadata, max_workers, batch_size, output, region):
    """Unified embed and store vectors command."""

    console = ctx.obj['console']
    cfg = ctx.obj['cfg']
    debug = ctx.obj.get('debug', False)
    region = get_region(region)

    multi_images = None
    batch_text_url = None

    # Load model properties once at start
    model = get_model_info(model_id)
    if not model:
        raise click.ClickException(f"Unsupported model: {model_id}")

    # Parse parameters
    user_dash_scope_params = {}
    if dashscope_inference_params:
        try:
            user_dash_scope_params = json.loads(dashscope_inference_params)
        except json.JSONDecodeError:
            raise click.ClickException("Invalid JSON in --dashscope-inference-params parameter")

    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise click.ClickException("Invalid JSON in --metadata parameter")

    # Early validation of user parameters before any processing
    if user_dash_scope_params:
        try:
            content_type = determine_content_type(text_value, text, image, video, multi_images, batch_text_url)
            system_keys = model.get_system_keys(content_type)
            system_payload = {key: None for key in system_keys}  # Dummy values for validation

            # Validate using utility function
            validate_user_parameters(system_payload, user_dash_scope_params)
        except ValueError as e:
            raise click.ClickException(str(e))

    # Check async model requirements
    if model.is_async() and not batch_text_url:
        raise click.ClickException(
            f"Async models like {model.model_id} require --batch-text-url parameter."
        )

    if not multi_images or len(multi_images) == 0:
        multi_images = None

    # Validate inputs
    is_multimodal = _validate_inputs(text_value, text, image, video, multi_images, model, key, filename_as_key, batch_text_url)

    try:
        # Initialize services
        dash_scope_service = DashScopeService(cfg, region, debug=debug, console=console)
        oss_vector_service = OSSVectorService(cfg, region, debug=debug, console=console)

        # Create unified processor
        processor = UnifiedProcessor(dash_scope_service, oss_vector_service, cfg)

        # Fetch index dimensions once at the top level
        try:
            index_info = oss_vector_service.get_index(vector_bucket_name, index_name)
            index_dimensions = index_info.get("dimension")
            if not index_dimensions:
                raise click.ClickException(f"Could not determine dimensions for index {index_name}")
        except Exception as e:
            raise click.ClickException(f"Failed to get index information: {str(e)}")

        # Prepare processing input
        processing_input = prepare_processing_input(
            text_value, text, image, video, multi_images, is_multimodal, metadata_dict, key, filename_as_key, key_prefix, batch_text_url
        )

        # Check if processing_input is None
        if processing_input is None:
            raise click.ClickException("No valid input provided. Please specify one of: --text-value, --text, --image, --video, --multi-images, or --batch-text-url")

        # Check for wildcard patterns (streaming batch processing)
        # Only process streaming batch if we have file_path and not batch_text_url
        if (processing_input.content_type in ["text", "image", "video", "multi_images"] and
            "file_path" in processing_input.data and
            "batch_text_url" not in processing_input.data):
            file_path = processing_input.data["file_path"]
            if '*' in file_path or '?' in file_path:
                return _process_streaming_batch(
                    file_path, processing_input.content_type, vector_bucket_name, index_name,
                    model, metadata_dict, user_dash_scope_params, batch_text_url,
                    processor, console, output, max_workers, batch_size, index_dimensions, processing_input.filename_as_key, processing_input.key_prefix
                )

        with _create_progress_context(console) as progress:
            # Process with unified pipeline
            progress.add_task("Processing input...", total=None)
            result = processor.process(
                model, processing_input, user_dash_scope_params,
                batch_text_url, vector_bucket_name, index_name, index_dimensions
            )

            # Store vectors with batch_size handling
            progress.add_task(f"Storing {len(result.vectors)} vector(s)...", total=None)

            # Handle batch_size for single file processing too
            vector_count = len(result.vectors)

            if vector_count <= batch_size:
                stored_keys = processor.store_vectors(result.vectors, vector_bucket_name, index_name)
            else:
                stored_keys = []
                for i in range(0, vector_count, batch_size):
                    chunk = result.vectors[i:i + batch_size]
                    chunk_keys = processor.store_vectors(chunk, vector_bucket_name, index_name)
                    stored_keys.extend(chunk_keys)

        # Prepare output
        if result.result_type == "multiclip":
            output_result = {
                'type': 'multiclip',
                'bucket': vector_bucket_name,
                'index': index_name,
                'model': model.model_id,
                'contentType': processing_input.content_type,
                'totalVectors': len(stored_keys),
                'keys': stored_keys
            }
            if result.job_id:
                output_result['jobId'] = result.job_id
        else:
            output_result = {
                'key': stored_keys[0],
                'bucket': vector_bucket_name,
                'index': index_name,
                'model': model.model_id,
                'contentType': processing_input.content_type,
                'embeddingDimensions': index_dimensions,
                'metadata': result.vectors[0]['metadata']
            }

        console.print_json(data=output_result)

    except Exception as e:
        raise click.ClickException(str(e))


if __name__ == '__main__':
    embed_put()


def _process_streaming_batch(file_path, content_type, vector_bucket_name, index_name,
                             model, metadata_dict, user_dash_scope_params, batch_text_url,
                             processor, console, output, max_workers, batch_size, index_dimensions, filename_as_key, key_prefix):
    """Process wildcard pattern using efficient streaming batch orchestrator."""

    try:
        # Create streaming batch orchestrator
        streaming_orchestrator = StreamingBatchOrchestrator(processor, max_workers, batch_size)

        console.print(f"Starting streaming batch processing: {file_path}")

        # Process using streaming approach (no pre-loading of file paths)
        batch_result = streaming_orchestrator.process_streaming_batch(
            file_path, content_type, vector_bucket_name, index_name, model,
            metadata_dict, batch_text_url, user_dash_scope_params, index_dimensions, filename_as_key, key_prefix
        )

        # Display results
        result_dict = {
            "type": "streaming_batch",
            "bucket": vector_bucket_name,
            "index": index_name,
            "model": model.model_id,
            "contentType": content_type,
            "totalFiles": batch_result.processed_count + batch_result.failed_count,
            "processedFiles": batch_result.processed_count,
            "failedFiles": batch_result.failed_count,
            "totalVectors": len(batch_result.processed_keys),  # Count actual vectors
            "vectorKeys": batch_result.processed_keys[:10] if batch_result.processed_keys else []  # Show first 10
        }

        if batch_result.errors:
            result_dict["errors"] = batch_result.errors[:10]  # Show first 10 errors

        if output == "table":
            _display_batch_table(result_dict, console)
        else:
            console.print_json(data=result_dict)

        # Print display limit messages after output
        if len(batch_result.processed_keys) > 10:
            console.print(f"[dim]Note: Showing first 10 of {len(batch_result.processed_keys)} vector keys[/dim]")

        if batch_result.errors and len(batch_result.errors) > 10:
            console.print(f"[dim]Note: Showing first 10 of {len(batch_result.errors)} errors[/dim]")

        return result_dict if output == "json" else None

    except Exception as e:
        console.print(f"[red]Streaming batch processing failed: {str(e)}[/red]")
        raise click.ClickException(f"Streaming batch processing failed: {str(e)}")

        # Prepare output in unified format
        result = {
            "type": "batch",
            "bucket": vector_bucket_name,
            "index": index_name,
            "model": model.model_id,
            "contentType": content_type,
            "totalFiles": batch_result.total_files,
            "processedFiles": batch_result.processed_count,  # Note: processed_count not processed_files
            "failedFiles": batch_result.failed_count,
            "totalVectors": batch_result.processed_count,  # Each file = 1 vector
            "vectorKeys": batch_result.processed_keys
        }

        if output == "table":
            _display_batch_table(result, console)
        else:
            console.print_json(data=result)

    except Exception as e:
        raise click.ClickException(f"Failed to process batch {file_path}: {str(e)}")


def _display_batch_table(result, console):
    """Display batch results in table format."""
    from rich.table import Table

    table = Table(title="Batch Processing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Files", str(result["totalFiles"]))
    table.add_row("Processed Files", str(result["processedFiles"]))
    table.add_row("Failed Files", str(result["failedFiles"]))
    table.add_row("Total Vectors", str(result["totalVectors"]))
    table.add_row("Model", result["model"])
    table.add_row("Content Type", result["contentType"])

    console.print(table)
