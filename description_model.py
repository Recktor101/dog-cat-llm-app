from transformers import pipeline

# Load Hugging Face Flan-T5 model
generator = pipeline("text2text-generation", model="google/flan-t5-base")

def get_breed_description(breed):
    prompt = f"Give a short and friendly description about the {breed} dog breed."
    result = generator(prompt, max_length=50)[0]['generated_text']
    return result
