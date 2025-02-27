from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

class TrainerHandler:
    def __init__(self, model, tokenizer, dataset, max_seq_length: int):
        """
        Initialize the Trainer Handler.

        Parameters:
            model: The model to train.
            tokenizer: Tokenizer associated with the model.
            dataset: Training dataset.
            max_seq_length (int): Maximum sequence length for training.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.max_seq_length: int = max_seq_length
        self.trainer = None

    def create_trainer(self):
        """
        Create and return the SFTTrainer instance.
        """
        self.trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            dataset_num_proc=1,
            packing=False,
            args=TrainingArguments(
                per_device_train_batch_size=2,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                max_steps=60,
                learning_rate=2e-4,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                report_to="none",
            ),
        )

    def train_model(self):
        """
        Train the model using the initialized trainer.
        """
        if self.trainer is None:
            raise ValueError("Trainer must be created before training the model.")

        trainer_stats = self.trainer.train()
        print("Training completed. Trainer stats:")
        print(trainer_stats)

        return trainer_stats
