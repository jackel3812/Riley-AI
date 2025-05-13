import os
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Replace the invalid model ID with a valid one
default_model_id = "gpt2"
model_id = os.getenv("MODEL_ID", default_model_id)

# Update the tokenizer and model initialization
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def ask_riley(prompt):
    if "[User]:" not in prompt:
        prompt = f"[User]: {prompt}"
    response = generator(prompt, max_length=512, truncation=True, do_sample=True, temperature=0.6)
    return response[0]["generated_text"]