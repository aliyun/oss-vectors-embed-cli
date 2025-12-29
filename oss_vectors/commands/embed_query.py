import json
from typing import Optional, Dict, Any

import click
from rich.console import Console
from rich.table import Table
from oss_vectors.core.unified_processor import UnifiedProcessor
from oss_vectors.core.services import DashScopeService, OSSVectorService
from oss_vectors.utils.config import get_region, setup_oss_cfg
from oss_vectors.utils.models import get_model_info, validate_model_modality, prepare_processing_input, determine_content_type


def _validate_query_inputs(query_input, text_value, text, image, video, multi_images, model, batch_text_url):
    """Validate query input parameters."""
    inputs = [query_input, text_value, text, image, video, multi_images, batch_text_url]
    provided_inputs = [
        inp for inp in inputs 
        if inp is not None and (not hasattr(inp, "__len__") or len(inp) > 0)
    ]

    if len(provided_inputs) == 0:
        raise click.ClickException(
            "No query input provided. Use one of: --text-value, --text, --image, --video, or --multi_images"
        )

    # Handle deprecated --query-input parameter
    if query_input:
        raise click.ClickException(
            "--query-input is deprecated and no longer supported. Use --text-value, --text, --image, --video, or --multi_images instead."
        )

    if not multi_images or len(multi_images) == 0:
        multi_images = None

    # Special case: Allow multimodal input for supported models
    is_multimodal_input = model.supports_multimodal_input() and sum(
        1 for inp in [text_value, text, image, video, multi_images] if inp is not None
    ) >= 2

    if len(provided_inputs) > 1 and not is_multimodal_input:
        raise click.ClickException(
            "Multiple query inputs provided. Use only one input type, except for multimodal queries with supported models (--text-value + --image)"
        )

    return is_multimodal_input


def _format_query_results(results: Dict[str, Any], output_format: str, console: Console):
    """Format and display query results."""
    if output_format == "table":
        _display_results_table(results, console)
    else:
        console.print_json(data=results)


def _display_results_table(results: Dict[str, Any], console: Console):
    """Display query results in table format."""
    table = Table(title="Query Results")
    table.add_column("Rank", style="cyan")
    table.add_column("Vector Key", style="green")
    table.add_column("Distance", style="yellow")
    table.add_column("Metadata", style="blue")

    query_results = results.get("results", [])
    for i, result in enumerate(query_results, 1):
        key = result.get("Key", "N/A")
        distance = f"{result.get('distance', 0):.4f}" if result.get('distance') is not None else "N/A"
        metadata = json.dumps(result.get("metadata", {}), indent=2) if result.get("metadata") else "None"

        table.add_row(str(i), key, distance, metadata)

    console.print(table)

    # Display summary
    summary = results.get("summary", {})
    console.print(f"\nQuery Summary:")
    console.print(f"  Model: {summary.get('model', 'N/A')}")
    console.print(f"  Results Found: {summary.get('resultsFound', 0)}")
    console.print(f"  Query Dimensions: {summary.get('queryDimensions', 'N/A')}")


@click.command()
@click.option('--vector-bucket-name', required=True, help='OSS bucket name for vector storage')
@click.option('--index-name', required=True, help='Vector index name')
@click.option('--model-id', required=True, help='DashScope embedding model ID (e.g., text-embedding-v1, text-embedding-v2, text-embedding-v3, text-embedding-v4, multimodal-embedding-v1, tongyi-embedding-vision-flash, tongyi-embedding-vision-plus, qwen2.5-vl-embedding)')
@click.option('--query-input', help='[DEPRECATED] Query text or file path - use specific input types instead')
@click.option('--text-value', help='Direct text query string')
@click.option('--text', help='Text file path (local file or OSS URI)')
@click.option('--image', help='Image file path (local file or OSS URI)')
@click.option('--video', help='Video file path (local file)')
@click.option('--top-k', default=5, type=int, help='Number of results to return (default: 5)')
@click.option('--filter', 'filter_expr', help='Filter expression for results (JSON format with operators, e.g., \'{"$and": [{"category": "docs"}, {"version": "1.0"}]}\')')
@click.option('--return-distance', is_flag=True, help='Return similarity distances in results')
@click.option('--return-metadata/--no-return-metadata', default=True, help='Return metadata in results (default: true)')
@click.option('--dashscope-inference-params', help='JSON string with model-specific parameters matching DashScope API format (e.g., \'{"dimension": "2048"}\' or \'{"input_type": "search_query"}\')')
@click.option('--output', type=click.Choice(['table', 'json']), default='json', help='Output format (default: json)')
@click.option('--region', help='OSS region name (effective in OSS path mode)')
@click.pass_context
def embed_query(ctx, vector_bucket_name, index_name, model_id, query_input, text_value, text, image, video,
                top_k, filter_expr, return_distance, return_metadata,
                dashscope_inference_params, output, region):
    """Embed query input and search for similar vectors using UnifiedProcessor.

    \b
    SUPPORTED QUERY INPUT TYPES:
    • Direct text: --text-value "search for this text"
    • Local text file: --text /path/to/query.txt
    • Local image file: --image /path/to/image.jpg
    • OSS text file: --text oss://bucket/query.txt
    • OSS image file: --image oss://bucket/image.jpg
    • Video files: --video /path/to/video.mp4
    • Multi images files: --multi_images /path/to/multi_images.jpg

    \b
    SUPPORTED MODELS:
    • text-embedding-v1 (text queries, 1,536 dimensions)
    • text-embedding-v2 (text queries, 1536 dimensions)
    • text-embedding-v3 (text queries, 1024(default)/768/512/256/128/64 dimensions)
    • text-embedding-v4 (text queries, 2048/1536/1024(default)/768/512/256/128/64 dimensions)
    • multimodal-embedding-v1 (text, image, video queries, 1024 dimensions)
    • tongyi-embedding-vision-flash (text, image, video, multi_images queries, 768 dimensions)
    • tongyi-embedding-vision-plus (text, image, video, multi_images queries, 1152 dimensions)
    • qwen2.5-vl-embedding (text, image, video queries, 2048/1024/768/512 dimensions)

    \b
    FILTERING:
    • Use JSON format with Alibaba Cloud OSS Vectors API operators
    • Single condition: --filter '{"category": {"$eq": "documentation"}}'
    • Multiple conditions (AND): --filter '{"$and": [{"category": "docs"}, {"version": "1.0"}]}'
    • Multiple conditions (OR): --filter '{"$or": [{"category": "docs"}, {"category": "guides"}]}'
    • Comparison operators: $eq, $ne, $gt, $gte, $lt, $lte, $in, $nin
    """
    console = Console()
    cfg = ctx.obj['cfg']
    
    multi_images = None
    batch_text_url = None

    try:
        # Get model information first for validation
        model = get_model_info(model_id)
        if not model:
            raise click.ClickException(f"Unsupported model: {model_id}")

        # Validate inputs
        is_multimodal = _validate_query_inputs(query_input, text_value, text, image, video, multi_images, model, batch_text_url)

        # Determine content type for model validation
        content_type = determine_content_type(text_value, text, image, video, multi_images, is_multimodal, batch_text_url)

        # Validate model capabilities
        if is_multimodal:
            if not model.supports_multimodal_input():
                raise click.ClickException(f"Model {model_id} does not support multimodal input (text + image)")
        else:
            validate_model_modality(model_id, content_type)

        # Validate async model requirements
        if model.is_async() and not batch_text_url:
            raise click.ClickException(
                f"Async models like {model.model_id} require --batch-text-url parameter."
            )

        # Parse user parameters
        user_dash_scope_params = {}
        if dashscope_inference_params:
            try:
                user_dash_scope_params = json.loads(dashscope_inference_params)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"Invalid JSON in --dashscope-inference-params: {e}")

        # Parse filter expression
        metadata_filter = None
        if filter_expr:
            try:
                metadata_filter = json.loads(filter_expr)
            except json.JSONDecodeError as e:
                raise click.ClickException(f"Invalid JSON in --filter: {e}")

        region = get_region(region)

        dash_scope_service = DashScopeService(cfg, region, debug=ctx.obj.get('debug', False), console=console)
        oss_vector_service = OSSVectorService(cfg, region, debug=ctx.obj.get('debug', False), console=console)

        # Create UnifiedProcessor
        processor = UnifiedProcessor(dash_scope_service, oss_vector_service, cfg)

        # Fetch index dimensions once at the top level (same pattern as PUT)
        try:
            index_info = oss_vector_service.get_index(vector_bucket_name, index_name)
            index_dimensions = index_info.get("dimension")
            if not index_dimensions:
                raise click.ClickException(f"Could not determine dimensions for index {index_name}")
        except Exception as e:
            raise click.ClickException(f"Failed to get index information: {str(e)}")

        # Create ProcessingInput (no metadata, keys, or object naming for queries)
        processing_input = prepare_processing_input(text_value, text, image, video, multi_images, is_multimodal, None, None, False, None, batch_text_url)

        # Process query input to generate embedding (same as PUT)
        with console.status("[bold green]Generating query embedding..."):
            # Validate query parameters for video/multi_images
            if model.is_async() and processing_input.content_type in ["video", "multi_images"]:
                if not user_dash_scope_params:
                    user_dash_scope_params = {}

                # Validate required parameters for video/multi_images queries
                if "startSec" not in user_dash_scope_params or "lengthSec" not in user_dash_scope_params:
                    raise click.ClickException('Both start time (startSec) and length (lengthSec) are required in --dashscope-inference-params for video/multi_images queries. Example: --dashscope-inference-params \'{"startSec": 30.0, "lengthSec": 6.0, "embeddingOption": ["visual-text"]}\'')

                # Validate embeddingOption for video queries (multi_images auto-selects)
                if processing_input.content_type == "video" and "embeddingOption" not in user_dash_scope_params:
                    raise click.ClickException('embeddingOption is required for video queries. Specify exactly one: ["visual-text"], ["visual-image"], or ["multi_images"]. Example: --dashscope-inference-params \'{"startSec": 30.0, "lengthSec": 6.0, "embeddingOption": ["visual-text"]}\'')

                # Validate embeddingOption has exactly one value for video queries
                if processing_input.content_type == "video" and "embeddingOption" in user_dash_scope_params:
                    embedding_options = user_dash_scope_params["embeddingOption"]
                    if not isinstance(embedding_options, list) or len(embedding_options) != 1:
                        raise click.ClickException('embeddingOption must contain exactly one value for video queries. Example: --dashscope-inference-params \'{"embeddingOption": ["visual-text"]}\'')

                # Calculate useFixedLengthSec from lengthSec if not explicitly provided
                if "useFixedLengthSec" not in user_dash_scope_params:
                    user_dash_scope_params["useFixedLengthSec"] = user_dash_scope_params["lengthSec"]

                # Validate useFixedLengthSec range
                use_fixed_length = user_dash_scope_params.get("useFixedLengthSec")
                if use_fixed_length is not None and (use_fixed_length < 2 or use_fixed_length > 10):
                    raise click.ClickException(f"Length of the clip must be between 2-10 seconds, got: {use_fixed_length}")

            result = processor.process(
                model=model,
                processing_input=processing_input,
                user_dash_scope_params=user_dash_scope_params,
                batch_text_url=batch_text_url,
                precomputed_dimensions=index_dimensions
            )

        # Extract query embedding
        query_timing = {}  # Store timing info for summary

        # Capture timing info from raw result before vector processing
        if model.is_async() and processing_input.content_type in ["video", "multi_images"] and hasattr(result, 'raw_results'):
            if result.raw_results and len(result.raw_results) > 0:
                first_raw = result.raw_results[0]
                if "startSec" in first_raw:
                    query_timing["queryStartSec"] = first_raw["startSec"]
                if "endSec" in first_raw:
                    query_timing["queryEndSec"] = first_raw["endSec"]

        if hasattr(result, 'vectors') and result.vectors:
            # Get the embedding from the first vector
            first_vector = result.vectors[0]
            if "embedding" in first_vector:
                query_embedding = first_vector["embedding"]
            elif "data" in first_vector and "float32" in first_vector["data"]:
                query_embedding = first_vector["data"]["float32"]
            else:
                raise click.ClickException(f"No embedding found in result. Available keys: {list(first_vector.keys())}")
        else:
            raise click.ClickException("Failed to generate query embedding - no vectors returned")

        # Perform vector similarity search
        with console.status("[bold green]Searching for similar vectors..."):
            search_results = oss_vector_service.query_vectors(
                bucket_name=vector_bucket_name,
                index_name=index_name,
                query_embedding=query_embedding,
                top_k=top_k,
                filter_expr=json.dumps(metadata_filter) if metadata_filter else None,
                return_metadata=return_metadata,
                return_distance=return_distance
            )

        # Format results
        formatted_results = {
            "results": [
                {
                    "Key": result.get("vectorId", ""),
                    "distance": result.get("similarity", 0.0),
                    "metadata": result.get("metadata", {})
                }
                for result in search_results
            ],
            "summary": {
                "queryType": content_type,
                "model": model_id,
                "index": index_name,
                "resultsFound": len(search_results),
                "queryDimensions": len(query_embedding),
                **query_timing  # Add timing info
            }
        }

        # Add distances if requested (already included in results)
        if not return_distance:
            for result in formatted_results["results"]:
                result.pop("distance", None)

        # Display results
        _format_query_results(formatted_results, output, console)

    except Exception as e:
        raise click.ClickException(str(e))
