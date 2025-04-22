# Biomedical RAG Chatbot with Image Generation

A powerful Retrieval-Augmented Generation (RAG) chatbot specialized in biomedical queries, featuring advanced image generation and editing capabilities. The system combines local LLM processing with Stable Diffusion for generating and editing biomedical images.

## 🌟 Features

### 1. RAG-based Query Processing
- **Vector Database**: FAISS-based storage for efficient similarity search
- **PDF Processing**: Extracts and chunks content from biomedical PDFs
- **Local LLM Integration**: Uses TinyLlama for generating responses
- **Source Attribution**: Tracks and displays source documents for transparency

### 2. Image Generation
- **Stable Diffusion Integration**: Generates high-quality biomedical images
- **Customizable Parameters**: Control steps, guidance scale, and seed
- **Medical-Specific Enhancements**: Optimized prompts for medical imagery
- **History Tracking**: Maintains generated image history

### 3. Advanced Image Editing
- **InstructPix2Pix**: Natural language-based image editing
- **ControlNet**: Structure-aware image modifications
- **Image-to-Image**: Complete image transformations
- **Basic Adjustments**: Brightness, contrast, sharpness controls

### 4. User Interface
- **Streamlit-based**: Clean, intuitive web interface
- **Chat History**: Persistent conversation tracking
- **Image Gallery**: Browse and reuse generated images
- **Real-time Preview**: Immediate feedback for image edits

## 🔧 System Architecture

```plaintext
├── Data Processing Layer
│   ├── PDF Extraction (PyMuPDF)
│   ├── Text Chunking
│   └── Vector Storage (FAISS)
│
├── RAG System
│   ├── Retriever (Semantic Search)
│   └── Generator (TinyLlama)
│
├── Image Generation
│   ├── Stable Diffusion
│   ├── InstructPix2Pix
│   └── ControlNet
│
└── Web Interface (Streamlit)
    ├── Chat Interface
    ├── Image Controls
    └── Edit Tools
```

## 📁 Project Structure

```plaintext
.
├── pmc_pdfs/               # PDF document storage
├── src/
│   ├── data_ingestion/    # PDF processing and vectorization
│   │   ├── pdf_processor.py
│   │   └── vectorizer.py
│   ├── models/           # RAG implementation
│   │   ├── generator.py
│   │   ├── retriever.py
│   │   └── rag.py
│   └── image_generation/ # Image generation and editing
│       ├── img_generator.py
│       └── img_editor.py
├── app.py               # Streamlit application
└── requirements.txt     # Project dependencies
```

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd biomedical-rag-chatbot
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your biomedical PDFs in the `pmc_pdfs` directory.

5. Process PDFs and create the vector index:
```bash
python src/data_ingestion/create_index.py
```

6. Run the application:
```bash
streamlit run app.py
```

## 💻 Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for faster image generation)
- 16GB RAM minimum (32GB recommended)
- Storage space for models and vector index

### Hardware Requirements
- **Minimum**:
  - CPU: 4 cores
  - RAM: 16GB
  - Storage: 20GB free space
  - GPU: 4GB VRAM (for GPU acceleration)

- **Recommended**:
  - CPU: 8+ cores
  - RAM: 32GB
  - Storage: 50GB free space
  - GPU: 8GB+ VRAM (NVIDIA RTX series)

## 🛠️ Configuration

### Vector Store Settings
```python
# src/data_ingestion/vectorizer.py
CHUNK_SIZE = 1000  # Adjust text chunk size
INDEX_TYPE = "L2"  # FAISS index type
```

### Image Generation Settings
```python
# src/image_generation/img_generator.py
DEFAULT_STEPS = 30
DEFAULT_GUIDANCE = 7.5
IMAGE_SIZE = 512  # Default image size
```

## 🎯 Usage Guide

### 1. Chat Interface
- Type your biomedical query in the chat input
- Include words like "show", "generate", or "image" for visual content
- View source citations below responses

### 2. Image Generation
- Adjust generation parameters in the sidebar
- Use the seed value for reproducible results
- Generated images appear in chat and history

### 3. Image Editing
- Select editing mode:
  - **Basic**: Simple adjustments
  - **Instruction-Based**: Natural language editing
  - **ControlNet**: Structure-aware editing
  - **Image-to-Image**: Complete transformation
- Use sliders to fine-tune parameters
- Preview changes in real-time

### 4. Saving and Exporting
- Save edited images using sidebar controls
- Access image history in the gallery
- Export chat history if needed

## 🔒 Security Considerations

- No API keys stored in code
- Local LLM processing for data privacy
- PDF content stored securely in vector database
- User session data cleared on restart


##  Future Improvements

1. Planned Features:
   - Multi-GPU support
   - Batch processing for PDFs
   - Enhanced medical term recognition
   - Advanced image composition tools

2. Performance Optimizations:
   - Improved vector indexing
   - Model quantization options
   - Caching for frequent queries

3. UI Enhancements:
   - Custom visualization tools
   - Advanced filtering options
   - Collaborative features



##  Acknowledgments

- PubMed Central for biomedical articles
- Hugging Face for model implementations
- Streamlit for the web framework 