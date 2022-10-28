from transformers import GPT2Tokenizer, OPTForCausalLM

model = OPTForCausalLM.from_pretrained("facebook/opt-125m")
tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-125m")

prompt = "Hey, are you consciours? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

generate_ids = model.generate(inputs.input_ids, max_length=30)
print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])