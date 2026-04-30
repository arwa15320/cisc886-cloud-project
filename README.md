# CISC 886 — Cloud Computing Project
## Customer Support Chatbot: Fine-Tuning Llama 3.2 on AWS

**Student:** 25wqgk  
**Course:** CISC 886 — Cloud Computing  
**Institution:** Queen's University, School of Computing  
**Region:** us-east-1 (N. Virginia)

---

## Project Overview

This project builds a complete end-to-end cloud-based customer support chatbot on AWS. It covers infrastructure provisioning, data preprocessing with Apache Spark on EMR, LLM fine-tuning using LoRA/QLoRA, model deployment with Ollama on EC2, and a browser-based chat interface via OpenWebUI.

**Model:** `meta-llama/Llama-3.2-3B-Instruct`  
**Dataset:** Bitext Customer Support Dataset (26,872 samples)  
**Task:** Customer support chatbot fine-tuned to handle account, order, billing, and shipping queries

---

## Repository Structure

```
cisc886-cloud-project/
├── main.tf                              # Terraform: VPC, subnet, IGW, route table, security group
├── preprocessing_optimized.py           # PySpark preprocessing pipeline (EMR)
├── chatbot_fine_tuning_completed.ipynb  # Fine-tuning notebook (Unsloth + LoRA on Colab)
├── api_runner.py                        # Ollama model serving script
├── web_ui.py                            # OpenWebUI configuration script
└── README.md                            # This file
```

---

## Prerequisites

### Tools Required
- [Terraform](https://developer.hashicorp.com/terraform/downloads) v1.5+
- [AWS CLI](https://aws.amazon.com/cli/) configured with your credentials
- [Git](https://git-scm.com/)
- [Python 3.10+](https://www.python.org/)
- [Google Colab](https://colab.research.google.com/) (free tier with T4 GPU)
- [Ollama](https://ollama.com/) (installed on EC2)

### Accounts Required
- AWS account with IAM user credentials
- Hugging Face account with access to `meta-llama/Llama-3.2-3B-Instruct`
- GitHub account

### AWS Region
All resources are deployed in: **`us-east-1` (N. Virginia)**

---

## Replication Steps

### Phase 1 — Infrastructure (VPC & Networking)

```bash
# Clone this repository
git clone https://github.com/YOURUSERNAME/cisc886-cloud-project.git
cd cisc886-cloud-project

# Initialize and apply Terraform
terraform init
terraform apply
```

This creates:
- VPC (`NetID-25wqgk-vpc`) with CIDR `10.0.0.0/16`
- Public subnet (`10.0.1.0/24`) in `us-east-1a`
- Internet Gateway and Route Table
- Security Group with ports 22, 80, 11434, 3000

---

### Phase 2 — S3 Setup & Dataset Upload

```bash
# Create S3 bucket
aws s3 mb s3://netid-25wqgk-cloud-storage-project --region us-east-1

# Upload raw dataset
aws s3 cp Bitext_customer_support.csv s3://netid-25wqgk-cloud-storage-project/row\ data/

# Upload PySpark script
aws s3 cp preprocessing_optimized.py s3://netid-25wqgk-cloud-storage-project/scripts/
```

---

### Phase 3 — EMR Cluster & PySpark Preprocessing

1. Go to AWS Console → EMR → Create Cluster
2. Configuration:
   - **Name:** `25wqgk-cluster`
   - **EMR Release:** `emr-7.13.0`
   - **Applications:** Spark
   - **Primary node:** `m5.xlarge`
   - **Core node:** `m5.xlarge` (1 instance)
   - **VPC:** `NetID-25wqgk-vpc`
   - **Subnet:** `NetID-25wqgk-subnet`
   - **Service role:** `EMR_DefaultRole`
   - **Instance profile:** `EMR_EC2_DefaultRole`

3. Add Step:
   - **Type:** Spark application
   - **Script:** `s3://netid-25wqgk-cloud-storage-project/scripts/preprocessing_optimized.py`

4. After completion → **Terminate the cluster immediately**

**Output folders in S3:**
```
s3://netid-25wqgk-cloud-storage-project/output/
├── train/
├── val/
├── test/
├── eda_category_distribution/
├── eda_intent_distribution/
├── eda_token_lengths/
├── eda_split_counts/
└── eda_length_summary/
```

---

### Phase 4 — Fine-Tuning on Google Colab

1. Open `chatbot_fine_tuning_completed.ipynb` in Google Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type → T4 GPU)
3. Run all cells sequentially
4. The notebook will:
   - Load `meta-llama/Llama-3.2-3B-Instruct` via Unsloth
   - Apply QLoRA fine-tuning with the following hyperparameters:

| Hyperparameter | Value |
|---|---|
| Learning rate | 2e-4 |
| Batch size | 2 |
| Gradient accumulation | 4 |
| Epochs | 3 |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Max sequence length | 2048 |
| Optimizer | AdamW 8-bit |

5. Export the model to GGUF format:
```python
model.save_pretrained_gguf("customer_support_chatbot", tokenizer, quantization_method="q4_k_m")
```

---

### Phase 5 — EC2 Deployment with Ollama

```bash
# SSH into EC2 instance
ssh -i 25wqgk-key.pem ec2-user@YOUR-EC2-PUBLIC-IP

# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Upload your GGUF model to EC2
scp -i 25wqgk-key.pem customer_support_chatbot_q4_k_m.gguf ec2-user@YOUR-EC2-IP:~/

# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./customer_support_chatbot_q4_k_m.gguf
SYSTEM "You are a helpful customer support assistant."
EOF

# Load the model into Ollama
ollama create customer-support-bot -f Modelfile

# Run the model
ollama run customer-support-bot

# Test via curl
curl http://localhost:11434/api/generate -d '{
  "model": "customer-support-bot",
  "prompt": "How do I cancel my order?",
  "stream": false
}'
```

---

### Phase 6 — OpenWebUI Setup

```bash
# Install Docker
sudo yum install docker -y
sudo systemctl start docker
sudo usermod -aG docker ec2-user

# Run OpenWebUI
docker run -d \
  --network=host \
  -v open-webui:/app/backend/data \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

# Configure auto-start on reboot
sudo systemctl enable docker
```

Access the interface at: `http://YOUR-EC2-PUBLIC-IP:3000`

---

## AWS Cost Summary

| Service | Usage | Estimated Cost |
|---|---|---|
| EMR Cluster | 2x m5.xlarge, ~10 minutes | ~$0.05 |
| EC2 Instance | t3.medium, ~2 hours | ~$0.04 |
| S3 Storage | ~500 MB total | ~$0.01 |
| Data Transfer | Minimal | ~$0.01 |
| **Total** | | **~$0.11** |

---

## References

- [Unsloth Fine-Tuning Guide](https://unsloth.ai/docs/get-started/fine-tuning-llms-guide/tutorial-how-to-finetune-llama-3-and-use-in-ollama)
- [Bitext Dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)
- [Llama 3.2 Model](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- [Ollama Documentation](https://ollama.com/docs)
- [OpenWebUI Documentation](https://docs.openwebui.com/)
