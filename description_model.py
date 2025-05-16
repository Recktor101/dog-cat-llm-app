from transformers import pipeline

# Hugging Face summarization model (you can replace with a better model later)
description_generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct", device=0 if torch.cuda.is_available() else -1)

def generate_description(breed):
    prompt = f"Describe the breed '{breed}' in detail, including its size, personality, and ideal environment."
    result = description_generator(prompt, max_new_tokens=60, do_sample=True)[0]['generated_text']
    return result.strip()
