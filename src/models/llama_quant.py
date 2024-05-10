from ctransformers import AutoModelForCausalLM

from .Model import Model


class LlamaQuant(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]
        self.gpu_layers = config["params"]["gpu_layers"]

        self.model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Llama-2-7b-Chat-GGUF", 
            model_file="llama-2-7b-chat.Q5_K_M.gguf", 
            model_type="llama", 
            gpu_layers=self.gpu_layers)

    def query(self, msg):
        return self.model(msg)
    
