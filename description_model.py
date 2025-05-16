from transformers import pipeline

generator = pipeline("text2text-generation", model="google/flan-t5-large")

def generate_breed_description(breed_name):
    prompt = f"Give a detailed description of the {breed_name} dog breed, including appearance, personality, and care tips."
    output = generator(prompt, max_length=150, do_sample=False)[0]["generated_text"]
    return output
