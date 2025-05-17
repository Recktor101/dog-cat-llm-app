from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import gc

# ✅ Use a small model to avoid memory issues on Streamlit Cloud
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_breed_description(animal_type, breed):
    """
    Generates a breed description for the given animal type (dog or cat) and breed.
    """
    if animal_type == "Dog":
        prompt = f"Write a short and informative description about the {breed} dog breed, including its personality, care needs, and unique traits."
    elif animal_type == "Cat":
        prompt = f"Write a short and informative description about the {breed} cat breed, including its personality, care needs, and unique traits."
    else:
        return "Unknown animal type."

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate the output
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=250,       # Allow longer responses
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode the output
    description = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ Free memory
    gc.collect()
    torch.cuda.empty_cache()

    return description.strip()
