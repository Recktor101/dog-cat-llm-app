from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
model_name = "google/flan-t5-base"  # You can switch to "google/flan-t5-large" if you want more power
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_breed_description(animal, breed):
    animal = animal.lower()
    
    if animal == "dog":
        prompt = (
            f"Write a short and informative description about the {breed} dog breed, "
            "including its personality, care needs, and unique traits."
        )
    elif animal == "cat":
        prompt = (
            f"Write a short and informative description about the {breed} cat breed, "
            "including its personality, care needs, and unique traits."
        )
    else:
        return "Unknown animal type."

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=250,        # Allow up to 250 tokens for detailed descriptions
        do_sample=True,            # Use sampling for more natural text
        temperature=0.7,           # Creativity balance
        top_k=50,                  # Top-k sampling to limit options
        top_p=0.95,                # Nucleus sampling
        eos_token_id=tokenizer.eos_token_id,
    )

    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description.strip()
