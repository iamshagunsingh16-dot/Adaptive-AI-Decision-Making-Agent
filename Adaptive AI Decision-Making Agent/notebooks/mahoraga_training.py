if not model or not model.has_module('q_proj'):
    raise ValueError('Model not loaded correctly or LoRA module q_proj not found in the model')
model = apply_lora(model)