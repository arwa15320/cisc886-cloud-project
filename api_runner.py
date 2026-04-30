from fastapi import FastAPI
from pydantic import BaseModel
from unsloth import FastLanguageModel
import torch
import re


# -----------------------------
# Model configuration
# -----------------------------

MODEL_PATH = "customer_support_chatbot_lora_emr"
MAX_SEQ_LENGTH = 2048

print("Loading fine-tuned model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

print("Model loaded successfully.")


# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(
    title="Customer Support Chatbot API",
    description="FastAPI LLM runner serving a fine-tuned Unsloth LoRA customer-support chatbot.",
    version="1.0.0",
)


SYSTEM_PROMPT = (
    "You are a helpful customer support chatbot. "
    "Answer politely, clearly, and directly. "
    "If the user provides a real order number, tracking number, shipment number, "
    "email, invoice number, or account detail, use that exact value in your answer. "
    "Only use placeholders such as {{Order Number}} or {{Tracking Number}} if the user explicitly typed that placeholder. "
    "Never replace a real customer-provided value with a placeholder. "
    "Do not invent missing customer details."
)


# -----------------------------
# Request schema
# -----------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 300


class GenerateResponse(BaseModel):
    model: str
    prompt: str
    response: str


# -----------------------------
# Helper functions
# -----------------------------

def extract_customer_values(message: str) -> dict:
    values = {}

    tracking_patterns = [
        r"(?:tracking number|tracking no|tracking id)\s*[:#-]?\s*([A-Za-z0-9\-]+)",
        r"(?:shipment number|shipment no|shipment id)\s*[:#-]?\s*([A-Za-z0-9\-]+)",
        r"(?:track|tracking|shipment)\s+(?:my\s+)?(?:shipment\s+)?([A-Za-z0-9\-]+)$",
        r"(?:shipment)\s*[:#-]?\s*([A-Za-z0-9\-]+)",
    ]

    order_patterns = [
        r"(?:order number|order no|order id)\s*[:#-]?\s*([A-Za-z0-9\-]+)",
        r"(?:cancel order|order)\s*[:#-]?\s*([A-Za-z0-9\-]+)",
    ]

    for pattern in tracking_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if not value.startswith("{{"):
                values["tracking_number"] = value
            break

    for pattern in order_patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if not value.startswith("{{"):
                values["order_number"] = value
            break

    return values


def replace_placeholders(response: str, values: dict) -> str:
    if "tracking_number" in values:
        response = response.replace("#{{Tracking Number}}", values["tracking_number"])
        response = response.replace("{{Tracking Number}}", values["tracking_number"])
        response = response.replace("#{{Tracking number}}", values["tracking_number"])
        response = response.replace("{{Tracking number}}", values["tracking_number"])

    if "order_number" in values:
        response = response.replace("#{{Order Number}}", values["order_number"])
        response = response.replace("{{Order Number}}", values["order_number"])
        response = response.replace("#{{Order number}}", values["order_number"])
        response = response.replace("{{Order number}}", values["order_number"])

    return response


def clean_response(decoded_text: str) -> str:
    if "<|im_start|>assistant" in decoded_text:
        return decoded_text.split("<|im_start|>assistant")[-1].strip()

    if "assistant" in decoded_text:
        return decoded_text.split("assistant")[-1].strip()

    return decoded_text.strip()


def generate_answer(user_prompt: str, max_new_tokens: int = 300) -> str:
    detected_values = extract_customer_values(user_prompt)

    extra_context = ""
    if detected_values:
        extra_context = (
            "\n\nImportant customer-provided values:\n"
            + "\n".join([f"- {key}: {value}" for key, value in detected_values.items()])
            + "\nUse these exact values in your answer. Do not replace them with placeholders."
        )

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT + extra_context,
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    response = clean_response(decoded)
    response = replace_placeholders(response, detected_values)

    return response


# -----------------------------
# API routes
# -----------------------------

@app.get("/")
def root():
    return {
        "status": "running",
        "runner": "FastAPI + Unsloth",
        "model": MODEL_PATH,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    answer = generate_answer(
        user_prompt=request.prompt,
        max_new_tokens=request.max_new_tokens,
    )

    return GenerateResponse(
        model=MODEL_PATH,
        prompt=request.prompt,
        response=answer,
    )