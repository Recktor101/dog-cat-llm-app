from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_breed_description(breed, animal_type="dog"):
    # Ensure correct animal is used in the prompt
    animal_type = animal_type.lower()
    if animal_type not in ["dog", "cat"]:
        animal_type = "animal"  # fallback

    # Adjust prompt depending on the animal
    prompt = f"Write a short and informative description about the {breed} {animal_type} breed, including its personality, care needs, and unique traits."

    # Tokenize the input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate output
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode and return
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description.strip()
