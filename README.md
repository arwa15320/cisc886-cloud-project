# CISC 886 — Cloud Computing Project
## Customer Support Chatbot on AWS

**Student NetID:** 25wqgk  
**Course:** CISC 886 — Cloud Computing  
**Institution:** Queen's University, School of Computing  
**AWS Region:** us-east-1 (N. Virginia)

---

## Project Overview

This project builds a complete end-to-end cloud-based customer support chatbot on AWS. It includes infrastructure provisioning with Terraform, data preprocessing using Apache Spark on EMR, model fine-tuning using Unsloth with LoRA/QLoRA, model export to GGUF format, deployment on an AWS EC2 CPU instance using Ollama, and a browser-based Gradio chat interface.

The final deployed chatbot is served through Ollama on EC2 and accessed through a Gradio web interface.

**Served model name:** `customer-support-chatbot`  
**Deployment runner:** Ollama CPU  
**Web interface:** Gradio  
**Dataset:** Bitext Customer Support Dataset  
**Task:** Customer support chatbot for order, shipment, account, billing, and support queries

**Base model:** `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` fine-tuned with Unsloth LoRA/QLoRA



---

## Repository Structure

```text
cisc886-cloud-project/
├── main.tf                              # Terraform infrastructure file
├── preprocessing_optimized.py           # PySpark preprocessing pipeline for EMR
├── chatbot_fine_tuning_completed.ipynb  # Fine-tuning notebook using Unsloth + LoRA/QLoRA
├── api_runner.py                        # Earlier FastAPI/Unsloth testing runner
├── web_ui.py                            # Earlier Gradio/Unsloth testing UI
└── README.md                            # Project replication guide
