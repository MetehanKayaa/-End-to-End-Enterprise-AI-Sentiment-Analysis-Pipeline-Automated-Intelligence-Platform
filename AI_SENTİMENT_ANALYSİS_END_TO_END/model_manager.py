import torch
from hardware_preparation import gpu_preparation
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from hugging_face_authentication import hugging_face_auth


class ModelManager:
    """
    Manages loading and configuration of a HuggingFace causal language model.
    Dynamically supports NVIDIA CUDA (with optional 4-bit quantization),
    Apple Silicon MPS (Metal acceleration), and CPU execution.
    """

    DEFAULT_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, use_quantization: bool = True):
        # 1. Hugging Face oturumunu doğrula
        hugging_face_auth()
        
        self.model_id = model_id
        # 2. Donanımı dinamik algıla ('cuda', 'mps', 'cpu')
        self.device = gpu_preparation()

        # 3. Kuantizasyon kontrolü (BitsAndBytes sadece CUDA'da çalışır)
        if use_quantization and self.device != "cuda":
            print(f"Notice: 4-bit quantization disabled on {self.device.upper()}. Using native precision.")
            self.use_quantization = False
        else:
            self.use_quantization = use_quantization

        # 4. Tokenizer ve Modeli yükle
        self.tokenizer = self.loading_tokenizer()
        self.model = self.loading_model()

    def loading_tokenizer(self):
        """
        Loads the Hugging Face tokenizer and ensures pad_token is set.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def loading_model(self):
        """
        Loads the model with optimal precision and device mapping based on the active platform.
        """
        kwargs = {
            "trust_remote_code": True,
        }

        # Platforma göre konfigürasyon
        if self.device == "cuda":
            kwargs["device_map"] = "auto"
            kwargs["dtype"] = torch.float16
            if self.use_quantization:
                kwargs["quantization_config"] = self.quantization_configuration()
        elif self.device == "mps":
            # Mac M4 Apple Silicon (MPS)
            kwargs["dtype"] = torch.float16
        else:
            # CPU Fallback
            kwargs["dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

        # Mac MPS için tensörleri Metal birimine taşı
        if self.device == "mps":
            model = model.to("mps")

        return model

    def quantization_configuration(self):
        """
        Returns BitsAndBytes 4-bit configuration for CUDA devices.
        """
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    def get_pipeline(self, max_new_tokens: int = 128, temperature: float = 0.7, batch_size: int = 16):
        """
        Returns a Hugging Face text-generation pipeline ready for batch inference.
        """
        pipe_device = 0 if self.device == "cuda" else (self.device if self.device == "mps" else -1)
        
        return pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=pipe_device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            batch_size=batch_size,
            return_full_text=False
        )

    def __repr__(self) -> str:
        if self.device == "cuda":
            dev_label = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif self.device == "mps":
            dev_label = "Apple Silicon (MPS)"
        else:
            dev_label = "CPU"

        return (
            f"ModelManager("
            f"model_id='{self.model_id}', "
            f"device='{dev_label}', "
            f"quantized={self.use_quantization})"
        )


if __name__ == "__main__":
    manager = ModelManager()
    print(manager)