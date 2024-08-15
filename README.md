## LawBot: Fine-Tuning LLaMA 2 on the Constitution of Pakistan

LawBot is an AI-powered chatbot designed to assist with legal inquiries related to the Constitution of Pakistan. Leveraging the LLaMA-2-7b model from Meta, LawBot has been fine-tuned to provide accurate and contextually relevant explanations of constitutional articles.

## Features

- ## Fine-Tuned LLaMA-2-7b Model:
This project utilizes the LLaMA-2-7b model, a state-of-the-art language model developed by Meta. The model has been fine-tuned specifically on legal text derived from the Constitution of Pakistan. This specialization allows the model to accurately understand and generate responses related to constitutional law, making it an effective tool for legal professionals, students, and researchers who require detailed and precise insights into legal content.

- ## Instruction-Response Training:
The model has been trained using a custom dataset of instruction-response pairs, meticulously extracted from the Constitution of Pakistan. This training method allows the model to handle specific queries by providing contextually accurate and relevant responses. The training process focuses on enhancing the model’s ability to generate human-like responses to complex legal instructions, ensuring that it can assist in legal interpretation and analysis with high reliability.

- ## Low-Rank Adaptation (LoRA):
Low-Rank Adaptation (LoRA) is employed in this project to efficiently fine-tune the LLaMA-2-7b model. LoRA reduces the number of 
trainable parameters by introducing low-rank matrices into the model, specifically targeting the attention layers. This approach 
allows for significant computational savings while maintaining the model's performance. By only adjusting these smaller matrices, 
LoRA enables effective adaptation to the legal texts from the Constitution of Pakistan without the need for extensive computational 
resources. This makes fine-tuning feasible even on limited hardware, ensuring a balance between efficiency and accuracy.

- ## High-Performance:
The model supports inference using FP16 (16-bit floating-point precision), which is optimized for modern GPUs. This capability allows the model to perform faster computations, reducing latency during inference without compromising the quality of the responses. The use of FP16 ensures that the model runs efficiently on compatible hardware, making it suitable for deployment in real-time legal consultation tools or applications where quick and accurate information retrieval is crucial.

## Project Structure

 - ## Data Preparation:
   Extract and format data from the Constitution of Pakistan.
 - ## Model Fine-Tuning:
   Fine-tune the LLaMA-2-7b model using LoRA.
 - ## Model Evaluation:
   Test and validate the model's performance on unseen data.
 - ## Model Deployment:
   Merge and deploy the fine-tuned model for production use.

## Requirements
 - Python 3.8+
 - PyTorch
 - Hugging Face Transformers
 - PEFT (Parameter-Efficient Fine-Tuning)
 - Hugging Face Datasets
 - TensorBoard
 - CUDA (for GPU acceleration)

## Installation
Install the necessary packages:
  ```sh
  pip install torch transformers peft datasets tensorboard
  ```

## Data Preparation
Data was extracted and formatted from the Constitution of Pakistan into instruction-response pairs. These pairs were used to fine-tune the model.

## Fine-Tuning
The fine-tuning process is conducted using LoRA to efficiently adapt the LLaMA-2 model for the legal domain.

```sh
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# Model and Tokenizer Setup
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype=torch.float16, device_map="auto")

# Fine-Tuning Configuration
training_arguments = TrainingArguments(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    logging_steps=32,
    learning_rate=1e-4,
    fp16=True,
    num_train_epochs=2,
    evaluation_strategy="steps",
    eval_steps=32,
    warmup_ratio=0.05,
    save_strategy="epoch",
    output_dir="experiments",
    report_to="tensorboard",
    save_safetensors=True,
)

# LoRA Configuration
peft_config = LoraConfig(
    r=2,
    lora_alpha=4,
    lora_dropout=0.1,
    target_modules=["q_proj", "up_proj", "o_proj", "k_proj", "down_proj", "gate_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

# Trainer Setup
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    peft_config=peft_config,
    tokenizer=tokenizer,
    args=training_arguments,
)

# Training
trainer.train()
trainer.save_model()
```

## Model Deployment
After training, the model is merged and prepared for deployment. The following script demonstrates how to load and use the merged model:

```sh
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the merged model
model = AutoModelForCausalLM.from_pretrained("merged_LLaMa2_7B_Chat-finetuned", torch_dtype=torch.float16).to("cuda:0")

# Tokenizer setup
tokenizer = AutoTokenizer.from_pretrained("merged_LLaMa2_7B_Chat-finetuned")

# Example Inference
def summarize(model, text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.0001, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

instruction = "Explain the main points of Article 101."
summary = summarize(model, instruction)
print(summary)
```
     
