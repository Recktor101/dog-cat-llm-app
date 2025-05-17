from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model (Flan-T5 or similar)
model_name = "google/flan-t5-base"  # Switch to "google/flan-t5-large" if needed and you have resources
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def get_breed_description(breed):
    # Prompt instructs model to generate a short, informative breed description
    prompt = f"Write a short and informative description about the {breed} dog breed, including its personality, care needs, and unique traits."

    # Tokenize input prompt for the model
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate breed description with sampling and no repeated phrases
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=250,          # Allows longer output
        do_sample=True,              # Enables diversity/randomness in output
        temperature=0.7,             # Controls creativity level
        top_k=50,                   # Limits sampling to top-k tokens
        top_p=0.95,                 # Nucleus sampling threshold
        eos_token_id=tokenizer.eos_token_id,  # Stop when model outputs EOS token
        no_repeat_ngram_size=3,      # Prevents repeating any 3-token sequences
    )

    # Decode output tokens to string and strip whitespace
    description = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Post-processing to ensure output ends with a proper sentence punctuation
    if not description.endswith(('.', '!', '?')):
        # Find the last sentence-ending punctuation
        last_period = max(description.rfind('.'), description.rfind('!'), description.rfind('?'))
        if last_period != -1:
            # Cut the description at the last punctuation
            description = description[:last_period+1]
        else:
            # If no punctuation found, just add a period
            description += '.'

    return description
