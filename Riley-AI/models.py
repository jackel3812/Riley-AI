from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

model_id = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def ask_riley(prompt):
    if "[User]:" not in prompt:
        prompt = f"[User]: {prompt}"
    response = generator(prompt, max_length=512, truncation=True, do_sample=True, temperature=0.6)
    return response[0]["generated_text"]