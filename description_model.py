from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model (Flan-T5 or similar)
model_name = "google/flan-t5-base"  # You can switch to "google/flan-t5-large" if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_breed_description(breed):
    prompt = f"Write a short and informative description about the {breed} dog breed, including its personality, care needs, and unique traits."

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate a response with improved settings
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=250,          # Increased to prevent cut-off
        do_sample=True,              # Enables diversity in output
        temperature=0.7,             # Balanced creativity
        top_k=50,                    # Top-k sampling
        top_p=0.95,                  # Nucleus sampling
        eos_token_id=tokenizer.eos_token_id,  # Helps stop correctly
    )

    # Decode and return the response
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description.strip()
