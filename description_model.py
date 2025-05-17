def get_breed_description(animal, breed):
    if animal.lower() == 'cat':
        prompt = f"Write a short and informative description about the {breed} cat breed, including its personality, care needs, and unique traits."
    elif animal.lower() == 'dog':
        prompt = f"Write a short and informative description about the {breed} dog breed, including its personality, care needs, and unique traits."
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
        no_repeat_ngram_size=3,
    )
    description = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    if not description.endswith(('.', '!', '?')):
        last_period = max(description.rfind('.'), description.rfind('!'), description.rfind('?'))
        if last_period != -1:
            description = description[:last_period+1]
        else:
            description += '.'
    return description
