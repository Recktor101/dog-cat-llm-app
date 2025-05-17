from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_breed_description(animal_type, breed):
    animal_type = animal_type.lower().strip()
    breed = breed.strip()

    if animal_type == "dog":
        prompt = f"Write a short and informative description about the {breed} dog breed, including its personality, care needs, and unique traits."
    elif animal_type == "cat":
        prompt = f"Write a short and informative description about the {breed} cat breed, including its personality, care needs, and unique traits."
    else:
        return "Unknown animal type."

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
    )
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return description.strip()
