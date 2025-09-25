# AIVault
![AIVault Logo](./docs/logo/aivault-logo.png)

[![Python Package using Conda](https://github.com/vahidnourbakhsh/AIVault/actions/workflows/python-app.yml/badge.svg)](https://github.com/vahidnourbakhsh/AIVault/actions/workflows/python-app.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## About

**AIVault** is a comprehensive collection of artificial intelligence models, methods, examples, and tutorials. This repository provides practical implementations and educational resources for various AI techniques including Generative AI, Machine Learning, Deep Learning, Computer Vision, Natural Language Processing, and more.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/vahidnourbakhsh/AIVault.git
cd AIVault

# Create conda environment (choose based on your system)
# For macOS (including Apple Silicon):
conda env create -f environment-macos.yml

# For Linux with GPU support:
conda env create -f environment-gpu.yml

# For basic setup (all platforms):
conda env create -f environment.yml

# Activate environment
conda activate aivault

# Install the package in development mode
pip install -e .

# Run tests
pytest tests/
```

## 📂 Repository Structure

```
AIVault/
├── generative_ai/          # Generative AI models and techniques
│   ├── large_language_models/
│   ├── image_generation/
│   ├── text_to_speech/
│   └── multimodal/
├── machine_learning/       # Classical ML algorithms and implementations
│   ├── supervised_learning/
│   ├── unsupervised_learning/
│   └── reinforcement_learning/
├── deep_learning/          # Deep learning models and architectures
│   ├── neural_networks/
│   ├── architectures/
│   └── optimization/
├── computer_vision/        # Computer vision models and techniques
│   ├── image_classification/
│   ├── object_detection/
│   ├── segmentation/
│   └── image_processing/
├── nlp/                   # Natural Language Processing
│   ├── text_classification/
│   ├── named_entity_recognition/
│   ├── sentiment_analysis/
│   └── text_generation/
├── model_optimization/    # Model optimization techniques
│   ├── quantization/
│   ├── pruning/
│   └── distillation/
├── deployment/           # Model deployment strategies
│   ├── containerization/
│   ├── api_servers/
│   └── edge_deployment/
├── utilities/           # Utility functions and helpers
├── data/               # Sample datasets and data loaders
├── tests/             # Test suite
├── docs/              # Documentation and tutorials
└── examples/          # End-to-end examples and use cases
```

## 🎯 Key Features

### 🤖 Generative AI
- **Large Language Models**: Implementation guides for LLMs, fine-tuning, and prompt engineering
- **Image Generation**: Stable Diffusion, GANs, VAEs, and custom image generators
- **Text-to-Speech**: Voice synthesis and audio generation models
- **Multimodal Models**: Vision-language models and cross-modal generation

### 🧠 Machine Learning
- **Supervised Learning**: Classification, regression, and ensemble methods
- **Unsupervised Learning**: Clustering, dimensionality reduction, and anomaly detection
- **Reinforcement Learning**: Policy optimization, value functions, and RL environments

### 🔥 Deep Learning
- **Neural Network Fundamentals**: From perceptrons to transformers
- **Popular Architectures**: ResNet, BERT, GPT, Vision Transformer, and more
- **Optimization Techniques**: Advanced training strategies and hyperparameter tuning

### 👁️ Computer Vision
- **Image Classification**: CNNs, Vision Transformers, and transfer learning
- **Object Detection**: YOLO, R-CNN families, and real-time detection
- **Semantic Segmentation**: U-Net, DeepLab, and instance segmentation
- **Image Processing**: Traditional CV techniques and modern approaches

### 📝 Natural Language Processing
- **Text Classification**: Sentiment analysis, spam detection, and topic modeling
- **Named Entity Recognition**: Information extraction and entity linking
- **Text Generation**: Language modeling, summarization, and creative writing
- **Conversational AI**: Chatbots, dialogue systems, and question answering

### ⚡ Model Optimization & Deployment
- **Quantization**: Post-training and quantization-aware training
- **Model Pruning**: Structured and unstructured pruning techniques
- **Knowledge Distillation**: Teacher-student model compression
- **Deployment**: Docker containers, REST APIs, and edge deployment

## 📖 Tutorials & Examples

Each module includes:
- ✅ **Theory Documentation**: Mathematical foundations and concepts
- ✅ **Step-by-step Implementations**: Detailed Jupyter notebooks with explanations
- ✅ **Real-world Examples**: Practical use cases with sample data
- ✅ **Best Practices**: Industry standards and optimization tips
- ✅ **Visualization**: Interactive plots and model interpretability tools

## 🛠️ Installation

### Prerequisites

- Python 3.10 or higher
- Conda or Miniconda
- Git

### Platform-Specific Setup

**For macOS (including Apple Silicon M1/M2/M3):**
```bash
git clone https://github.com/vahidnourbakhsh/AIVault.git
cd AIVault
conda env create -f environment-macos.yml
conda activate aivault
```

**For Linux with NVIDIA GPU:**
```bash
git clone https://github.com/vahidnourbakhsh/AIVault.git
cd AIVault
conda env create -f environment-gpu.yml
conda activate aivault
```

**For any platform (basic setup):**
```bash
git clone https://github.com/vahidnourbakhsh/AIVault.git
cd AIVault
conda env create -f environment.yml
conda activate aivault
```

### Development Setup
```bash
# After creating the environment
pip install -e .

# Install pre-commit hooks
pre-commit install
```

**Note:** The macOS environment excludes CUDA-specific packages (like `bitsandbytes`, `flash-attn`, `vllm`) since CUDA is not available on macOS. Instead, it uses MPS (Metal Performance Shaders) for GPU acceleration on Apple Silicon.

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=aivault --cov-report=html

# Run specific test module
pytest tests/test_generative_ai.py
```

## 📊 Datasets

The `data/` directory contains:
- Sample datasets for tutorials
- Data loading utilities
- Preprocessing scripts
- Benchmark datasets for model evaluation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- Inspired by the amazing AI research community
- Built with popular frameworks: PyTorch, TensorFlow, Hugging Face, and more
- Special thanks to contributors and the open-source community

## 📞 Contact

- **Author**: Vahid Nourbakhsh
- **LinkedIn**: [https://www.linkedin.com/in/vahidnourbakhsh/](https://www.linkedin.com/in/vahidnourbakhsh/)

---

⭐ **Star this repository if you find it helpful!** ⭐
