from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = "I have 10 chocolates, I gave 3 of them to my friend. How many left??"
checkpoint = "openai-community/gpt2-large"



tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer.encode(prompt, return_tensors="pt")
attention_mask = torch.ones(inputs.shape, dtype=torch.long)


model = AutoModelForCausalLM.from_pretrained(checkpoint)

outputs = model.generate(inputs, attention_mask=attention_mask, num_beams=5, max_new_tokens=200,do_sample=True,no_repeat_ngram_size=2, early_stopping=True)
text=tokenizer.batch_decode(outputs, skip_special_tokens=True)
print(text)