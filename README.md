# Alibaba Cloud OSS Vectors Embed CLI

Alibaba Cloud OSS Vectors Embed CLI is a standalone command-line tool that simplifies the process of working with vector embeddings in OSS Vectors. You can create vector embeddings for your data using Alibaba Cloud DashScope and store and query them in your OSS vector index using single commands. 

**Alibaba Cloud OSS Vectors Embed CLI is in preview release and is subject to change.**

## Supported Commands

**oss-vectors-embed put**: Embed text, file content, or OSS objects and store them as vectors in an OSS vector index.
You can create and ingest vector embeddings into an OSS vector index using a single put command. You specify the data input you want to create an embedding for, an Alibaba Cloud DashScope embeddings model ID, your OSS vector bucket name, and OSS vector index name. The command supports several input formats including text data, a local text or image file, an OSS image or text object or prefix. The command generates embeddings using the dimensions configured in your OSS vector index properties. If you are ingesting embeddings for several objects in an OSS prefix or local file path, it automatically uses batch processes to maximize throughput. 

**Note**: Each file is processed as a single embedding. Document chunking is not currently supported. 

**oss-vectors-embed query**: Embed a query input and search for similar vectors in an OSS vector index.
You can perform similarity queries for vector embeddings in your OSS vector index using a single query command. You specify your query input, an Alibaba Cloud DashScope embeddings model ID, the vector bucket name, and vector index name. The command accepts several types of query inputs like a text string, an image file, or a single OSS text or image object. The command generates embeddings for your query using the input embeddings model and then performs a similarity search to find the most relevant matches. You can control the number of results returned, apply metadata filters to narrow your search, and choose whether to include similarity distance in the results for comprehensive analysis.

### Supported Input Types

**Note**: 
This CLI has introduced a unified `--dashscope-inference-params` parameter for all model-specific parameters.
Additionally, the query command uses the following separate parameters:

- **`--text-value`**: Direct text query string (preferred for text queries)
- **`--text`**: Text file path (local file or OSS URI)
- **`--image`**: Image file path (local file or OSS URI or URI)
- **`--video`**: Video file path (URI)

## Installation and Configuration
### Prerequisites
- Python 3.9 or higher
- To execute the CLI, you will need Alibaba Cloud credentials configured. 
- Update your Alibaba Cloud account with appropriate permissions to use Alibaba Cloud DashScope and OSS Vectors
- Access to an Alibaba Cloud DashScope embedding model
- Create an Alibaba Cloud OSS vector bucket and vector index to store your embeddings

### Quick Install (Recommended)
```bash
pip install oss-vectors-embed-cli
```

### Development Install
```bash
# Clone the repository
git clone https://github.com/aliyun/oss-vectors-embed-cli.git
cd oss-vectors-embed-cli

# Install in development mode
pip install -e .
```

**Note**: All dependencies are automatically installed when you install the package via pip.

### Quick Start

#### **Put Examples**

#### ** Examples for the Text-Embedding Model **
**Note:** There are four general text vector models: text-embedding-v1, text-embedding-v2, text-embedding-v3, and text-embedding-v4. Here, we use text-embedding-v4 as an example. 

1. **Embed text and store them as vectors in your OSS vector index:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text-value "Hello, world!"
```

2. **Process local text files:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text "./documents/sample.txt"
```

3. **Process files from a local file path using wildcard characters:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text "./documents/*.txt"
```

4. **Process files from an OSS general purpose bucket using wildcard characters:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --region cn-hangzhou \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text "oss://bucket/path/*"
```

5. **Add metadata alongside your vectors:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --region cn-hangzhou \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text "oss://my-bucket/sample.txt" \
  --metadata '{"category": "technology", "version": "1.0"}'
```

6. **Use custom model parameters:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text-value "Custom parameters" \
  --dashscope-inference-params '{"output_type": "dense", "dimension": "1024"}'
```

7. **Use custom vector key:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text-value "Custom vector key" \
  --key "text-1"
```

8. **Use OSS object key as vector key:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --region cn-hangzhou \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text "oss://my-bucket/sample.txt" \
  --filename-as-key
```

9. **Use filename as vector key for batch processing:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text "./documents/*.txt" \
  --filename-as-key
```

10. **Use key prefix with custom key:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text-value "Use key prefix with custom key" \
  --key "text-1" \
  --key-prefix "prefix-a/"
```

11. **Use key prefix with filename:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text "./documents/sample.txt" \
  --filename-as-key \
  --key-prefix "prefix-doc/"
```

#### ** Examples for the Multimodal-Embedding Model **
**Note:** There are four general multimodal vector models: multimodal-embedding-v1, tongyi-embedding-vision-flash, tongyi-embedding-vision-plus, and qwen2.5-vl-embedding. Here, we use qwen2.5-vl-embedding as an example. 

12. **Embed multimodal and store them as vectors in your OSS vector index:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id qwen2.5-vl-embedding \
  --text-value "Hello, world!"
```

13. **Process image files using a local file path:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id qwen2.5-vl-embedding \
  --image "./images/photo.jpg"
```

14. **Process video files using an url:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id qwen2.5-vl-embedding \
  --video "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20250107/lbcemt/new+video.mp4"
```

15. **Process files from a local file path using wildcard characters:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id qwen2.5-vl-embedding \
  --image "./documents/*.jpg"
```

16. **Process files from an OSS general purpose bucket using wildcard characters:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  put \
  --region cn-hangzhou \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id qwen2.5-vl-embedding \
  --image "oss://bucket/path/*"
```

#### **Query Examples**

1. **Direct text query:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text-value "query text" \
  --top-k 20
```

2. **Query using a local text file:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text "./documents/query.txt" \
  --top-k 20 \
  --output table
```

3. **Query using an OSS text file:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  query \
  --region cn-hangzhou \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text "oss://my-bucket/query.txt" \
  --top-k 20 
```

4. **Image query:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id qwen2.5-vl-embedding \
  --image "./documents/image.jpg" \
  --top-k 20 
```

5. **Text: Query with metadata filters:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text-value "query text" \
  --filter '{"category": {"$eq": "technology"}}' \
  --top-k 20 \
  --return-metadata
```
6. **Text: Query with multiple metadata filters (AND):**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text-value "query text" \
  --filter '{"$and": [{"category": "technology"}, {"version": "1.0"}]}' \
  --top-k 20 \
  --return-metadata
```

7. **Text: Query with multiple metadata filters (OR):**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text-value "query text" \
  --filter '{"$or": [{"category": "docs"}, {"category": "guides"}]}' \
  --top-k 20
```

8. **Text: Query with metadata filters (comparison operators):**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id text-embedding-v4 \
  --text-value "query text" \
  --filter '{"$and": [{"category": "tech"}, {"version": {"$eq": "1.0"}}]}' \
  --top-k 20
```

9. **Qwen2.5: Query with custom model parameters:**
```bash
oss-vectors-embed \
  --account-id 12***345 \
  --vectors-region cn-hangzhou \
  query \
  --vector-bucket-name my-bucket \
  --index-name my-index \
  --model-id qwen2.5-vl-embedding \
  --text-value "search query with custom truncation" \
  --dashscope-inference-params '{"truncate": "END"}' \
  --top-k 20 \
  --return-distance
```


### Command Parameters

#### Global Options
- `--debug`: Enable debug mode with detailed logging for troubleshooting
- `--account-id`: Alibaba Cloud account id
- `--vectors-region`: OSS vectors bucket region name (config defaults)
- `--vectors-endpoint`: The domain names that other services can use to access OSS vectors bucket

#### Put Command Parameters
Required:
- `--vector-bucket-name`: Name of the OSS vector bucket 
- `--index-name`: Name of the vector index in your vector index to store the vector embeddings
- `--model-id`: DashScope model ID to use for generating embeddings (e.g., text-embedding-v4, qwen2.5-vl-embedding)

Input Options (one required):
- `--text-value`: Direct text input to embed
- `--text`: Text input - supports multiple input types:
  - **Local file**: `./document.txt`
  - **Local files with wildcard characters**: `./data/*.txt`, `~/docs/*.md`
  - **OSS object**: `oss://bucket/path/file.txt`
  - **OSS path with wildcard characters**: `oss://bucket/path/*` (prefix-based, not extension-based)
- `--image`: Image input - supports multiple input types:
  - **Local file**: `./document.jpg`
  - **Local wildcard**: `./data/*.jpg`
  - **OSS object**: `oss://bucket/path/file.jpg`
  - **URI**: `https://path/pic.jpg` 
  - **OSS path with wildcard characters**: `oss://bucket/path/*` (prefix-based, not extension-based)
- `--video`: Video input - supports: 
  - **URI**: `https://path/video.mp4` 

Optional:
- `--region`: OSS region name (effective in OSS path mode)
- `--key`: Uniquely identifies each vector in the vector index (default: auto-generated UUID)
- `--key-prefix`: Prefix to prepend to all vector keys (works with --key, --filename-as-key, and auto-generated UUIDs)
- `--filename-as-key`: Use filename as vector key (mutually exclusive with --key)
- `--metadata`: Additional metadata associated with the vector; provided as JSON string
- `--dashscope-inference-params`: Model-specific parameters passed to DashScope (JSON format, e.g., `'{"dimension": "1024"}'`)
- `--max-workers`: Maximum parallel workers for batch processing (default: 4)
- `--batch-size`: Number of vectors per OSS Vector put_vectors call (1-500, default: 500)
- `--output`: Output format (json or table, default: json)



#### Query Command Parameters

**Core Required Parameters:**
- `--vector-bucket-name`: Name of the OSS vector bucket
- `--index-name`: Name of the vector index 
- `--model-id`:DashScope model ID to use for generating embeddings (e.g., text-embedding-v4, qwen2.5-vl-embedding)

**Query Input Parameters (One Required):**
- `--text-value`: Direct text query string
- `--text`: Text file path (local file or OSS URI)
- `--image`: Image file path (local file or OSS URI or URI)
- `--video`: Video file path (URI)


**Optional Parameters:**
- `--region`: OSS region name (effective in OSS path mode)
- `--top-k`: Number of results to return (default: 30)
- `--filter`: Filter expression for metadata-based filtering (JSON format with Alibaba Cloud OSS Vectors API operators)
- `--dashscope-inference-params`: Model-specific parameters passed to DashScope (JSON format, e.g., `'{"truncate": "END"}'`)
- `--return-metadata`: Include metadata in results (default: true)
- `--return-distance`: Include similarity distance scores
- `--output`: Output format (table or json, default: json)

**Query Examples:**

```bash
# Direct text query (preferred method)
oss-vectors-embed --account-id 12***345 --vectors-region cn-hangzhou query --vector-bucket-name my-bucket --index-name my-index \
  --model-id text-embedding-v4 --text-value "search text" --top-k 10

# Text file query
oss-vectors-embed --account-id 12***345 --vectors-region cn-hangzhou query --vector-bucket-name my-bucket --index-name my-index \
  --model-id text-embedding-v4 --text ./query.txt --top-k 5

# Image query
oss-vectors-embed --account-id 12***345 --vectors-region cn-hangzhou query --vector-bucket-name my-bucket --index-name my-index \
  --model-id text-embedding-v4 --image ./query-image.jpg --top-k 3
```
