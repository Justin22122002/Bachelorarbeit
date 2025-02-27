from unsloth import FastLanguageModel


class LanguageModelHandler:
    def __init__(self, model_path: str, max_seq_length: int =2048, dtype=None, load_in_4bit: bool =True):
        """
        Initialize the Language Model Handler.

        Parameters:
            model_path (str): Path to the pretrained model or Hugging Face identifier.
            max_seq_length (int): Maximum sequence length for the model.
            dtype: Data type for model parameters (e.g., torch.float16).
            load_in_4bit (bool): Whether to use 4-bit quantization for reduced memory usage.
        """
        self.model_path: str = model_path
        self.max_seq_length: int = max_seq_length
        self.dtype = dtype
        self.load_in_4bit: bool = load_in_4bit
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """
        Load the model and tokenizer from the specified path.
        """
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_path,
                max_seq_length=self.max_seq_length,
                dtype=self.dtype,
                load_in_4bit=self.load_in_4bit,
            )
            print("Model and tokenizer successfully loaded.")
        except Exception as e:
            print(f"Error loading model: {e}")

    def apply_peft(
            self,
            r=16,
            target_modules=None,
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None
    ):
        """
        Apply Parameter-Efficient Fine-Tuning (PEFT) to the loaded model.

        Parameters:
            r (int): Rank of the LoRA decomposition.
            target_modules (list): List of module names to apply LoRA.
            lora_alpha (int): Scaling factor for LoRA updates.
            lora_dropout (float): Dropout probability for LoRA layers.
            bias (str): Bias configuration for LoRA ("none", "all", etc.).
            use_gradient_checkpointing (bool or str): Enable gradient checkpointing for memory efficiency.
            random_state (int): Random seed for reproducibility.
            use_rslora (bool): Use rank-stabilized LoRA.
            loftq_config (any): Configuration for LoftQ, if applicable.
        """
        if self.model is None:
            raise ValueError("Model must be loaded before applying PEFT.")

        target_modules = target_modules or [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

        try:
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=r,
                target_modules=target_modules,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=bias,
                use_gradient_checkpointing=use_gradient_checkpointing,
                random_state=random_state,
                use_rslora=use_rslora,
                loftq_config=loftq_config,
            )
            print("PEFT applied successfully.")
        except Exception as e:
            print(f"Error applying PEFT: {e}")