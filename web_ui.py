from unsloth import FastLanguageModel
import gradio as gr
import torch
import re

MODEL_PATH = "customer_support_chatbot_lora_emr"
MAX_SEQ_LENGTH = 2048

print("======================================")
print("Starting Gradio web chat interface")
print(f"Fine-tuned model name: {MODEL_PATH}")
print("Interface: Gradio Chat UI + Unsloth")
print("======================================")

print("Loading fine-tuned model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

FastLanguageModel.for_inference(model)

print("Model loaded successfully.")
print(f"Web interface serving model: {MODEL_PATH}")

SYSTEM_PROMPT = (
    "You are a helpful customer support chatbot. "
    "Answer politely, clearly, and directly. "
    "If the user provides a real order number, tracking number, shipment number, "
    "email, invoice number, or account detail, use that exact value in your answer. "
    "Only use placeholders such as {{Order Number}} or {{Tracking Number}} if the user explicitly typed that placeholder. "
    "Never replace a real customer-provided value with a placeholder. "
    "Do not invent missing customer details."
)

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

def add_history_to_messages(messages: list, history) -> list:
    if not history:
        return messages

    for item in history:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, bot_msg = item
            if user_msg:
                messages.append({"role": "user", "content": str(user_msg)})
            if bot_msg:
                messages.append({"role": "assistant", "content": str(bot_msg)})

        elif isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": str(content)})

    return messages

def respond(message, history):
    detected_values = extract_customer_values(message)

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
        }
    ]

    messages = add_history_to_messages(messages, history)
    messages.append({"role": "user", "content": message})

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
            max_new_tokens=300,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = clean_response(decoded)
    response = replace_placeholders(response, detected_values)

    return response

demo = gr.ChatInterface(
    fn=respond,
    title="Customer Support Chatbot",
    description=f"Fine-tuned model: {MODEL_PATH} | Runner: Gradio + Unsloth",
    examples=[
        "I want to cancel my order 12345",
        "I need help tracking my shipment 20",
        "How can I change the email address on my account?",
        "I want a refund for my recent purchase",
    ],
)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )