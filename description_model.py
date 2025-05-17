from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model only once, outside the function
model_name = "google/flan-t5-base"  # or "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_breed_description(animal, breed):
    if animal.lower() == 'cat':
        prompt = f"Write a short and informative description about the {breed} cat breed, including its personality, care needs, and unique traits."
    elif animal.lower() == 'dog':
        prompt = f"Write a short and informative description about the {breed} dog breed, including its personality, care needs, and unique traits."
    else:
        return "Unknown animal type."

    inputs = tokenizer(prompt, return_tensors="pt")  # tokenizer must be defined

    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=250,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        eos_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
    )

    description = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Ensure description ends with a proper punctuation
    if not description.endswith(('.', '!', '?')):
        last_period = max(description.rfind('.'), description.rfind('!'), description.rfind('?'))
        if last_period != -1:
            description = description[:last_period+1]
        else:
            description += '.'

    return description
