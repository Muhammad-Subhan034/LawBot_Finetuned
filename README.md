LawBot: Fine-Tuning LLaMA 2 on the Constitution of Pakistan
Overview
This project involves fine-tuning the LLaMA 2 model, specifically the 7 billion parameter variant (LLaMA-2-7b), to create a chatbot tailored for legal inquiries related to the Constitution of Pakistan. The goal is to develop a model that can accurately and efficiently answer questions and provide summaries based on constitutional articles.

Project Structure
The project is organized into the following key stages:

Data Preparation: Extraction and formatting of text data from the Constitution of Pakistan.
Model Fine-Tuning: Training the LLaMA-2-7b model using LoRA (Low-Rank Adaptation) on the prepared dataset.
Model Evaluation: Assessing the performance of the fine-tuned model.
Model Deployment: Merging the LoRA-adapted model with the base model for deployment.
Features
Base Model: meta-llama/LLaMA-2-7b-hf
Data Source: Constitution of Pakistan
Fine-Tuning Technique: LoRA (Low-Rank Adaptation)
Training Framework: Hugging Face Transformers, PEFT (Parameter-Efficient Fine-Tuning)
Deployment: Model merging and inference setup
Requirements
Python 3.8+
PyTorch
Transformers
PEFT (Parameter-Efficient Fine-Tuning)
Hugging Face Datasets
Hugging Face Hub (for model authentication)
TensorBoard (for monitoring)
Data Preparation
The training data was extracted from the Constitution of Pakistan, formatted into instruction-response pairs. Each instruction corresponds to a legal question or a directive, while the response contains a summary or explanation of the relevant constitutional article.

Model Fine-Tuning
The fine-tuning process uses LoRA to adapt the LLaMA-2-7b model to the specific task. The training is conducted with careful attention to memory usage and model performance.

Model Evaluation and Inference
After fine-tuning, the model is merged with the base model to prepare it for deployment. The model can be used to generate summaries and explanations of constitutional articles.

Results
The fine-tuned model effectively summarizes and explains various articles from the Constitution of Pakistan. The evaluation metrics and examples of responses can be found in the experiments directory.
