from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
model_name = "google/flan-t5-base"  # You can also try flan-t5-large if you're running locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_breed_description(breed):
    # Adjust prompt to work for both dogs and cats
    prompt = f"Write a short and informative description about the {breed} animal breed, including its personality, care needs, and unique traits."

    # Tokenize the prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output with advanced generation settings
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=250,              # Longer descriptions
        do_sample=True,                  # Sampling for variety
        temperature=0.7,                 # Balanced randomness
        top_k=50,                        # Top-k filtering
        top_p=0.95,                      # Nucleus sampling
        eos_token_id=tokenizer.eos_token_id  # Proper stopping
    )

    # Decode and return the output
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description.strip()
