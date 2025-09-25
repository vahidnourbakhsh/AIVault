# AIVault Examples

This directory contains end-to-end examples and use cases demonstrating various AI techniques implemented in AIVault.

## Structure

- `getting_started/` - Basic tutorials for newcomers
- `generative_ai/` - Examples of generative AI applications  
- `computer_vision/` - Image processing and CV examples
- `nlp/` - Natural language processing examples
- `deployment/` - Model deployment examples
- `advanced/` - Advanced techniques and research implementations

## Running Examples

1. Make sure you have the AIVault environment set up:
```bash
conda env create -f environment.yml
conda activate aivault
```

2. Install AIVault in development mode:
```bash
pip install -e .
```

3. Navigate to any example directory and run the notebooks:
```bash
jupyter lab
```

## Example Categories

### Getting Started
- `hello_aivault.ipynb` - Introduction to AIVault
- `data_preprocessing.ipynb` - Data handling basics
- `model_training.ipynb` - Basic model training workflow

### Generative AI
- `text_generation_llm.ipynb` - Text generation with LLMs
- `image_generation_diffusion.ipynb` - Image generation with diffusion models
- `multimodal_ai.ipynb` - Cross-modal AI applications

### Computer Vision  
- `image_classification.ipynb` - Image classification tutorial
- `object_detection.ipynb` - Object detection examples
- `style_transfer.ipynb` - Neural style transfer

### Natural Language Processing
- `sentiment_analysis.ipynb` - Sentiment analysis pipeline
- `named_entity_recognition.ipynb` - NER implementation
- `text_summarization.ipynb` - Automatic text summarization

### Model Deployment
- `fastapi_deployment.ipynb` - REST API deployment
- `streamlit_app.ipynb` - Web app with Streamlit
- `docker_containerization.ipynb` - Docker deployment guide

Each example includes:
- Clear explanations of concepts
- Step-by-step implementation
- Visualization of results
- Best practices and tips
