from transformers import GPT2Tokenizer , GPT2LMHeadModel

# Tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
model = GPT2LMHeadModel.from_pretrained("gpt2-large", pad_token_id=tokenizer.eos_token_id) # Token Id conversion from {pad to eos}imp

# Sentence
sequence = "I have 10 chocolates, I gave 3 of them to my friend. How many left?"

# Encoding the sentence
load = tokenizer.encode(sequence,return_tensors='pt')

# generate text
result = model.generate(load,max_length=200,do_sample=True,num_beams=5,no_repeat_ngram_size=2,early_stopping=True)

# decoding text
text = tokenizer.decode(result[0], skip_special_token=True)
print(text)


