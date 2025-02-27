import torch

def formatting_prompts_func(examples, tokenizer, alpaca_prompt):
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN
    instructions = examples["instruction"]
    inputs = examples["input"]
    outputs = examples["output"]
    texts = []
    for instruction, _input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, _input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

# Funktion: Speicherstatistiken nach dem Training berechnen
def calculate_memory_after(trainer_runtime_seconds, start_memory, max_memory):
    """
    Berechnet GPU-Speicherstatistiken basierend auf dem Training und der GPU-Auslastung.

    :param trainer_runtime_seconds: Gesamte Trainingszeit in Sekunden.
    :param start_memory: Zu Beginn reservierter GPU-Speicher in GB.
    :param max_memory: Maximal verfügbarer GPU-Speicher in GB.
    :return: Dictionary mit Speicherstatistiken.
    """
    # Aktuelle Speicherwerte abrufen
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_memory, 3)

    # Prozentsätze berechnen
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    # Laufzeit in Minuten berechnen
    runtime_minutes = round(trainer_runtime_seconds / 60, 2)

    # Ergebnisse in einem Dictionary zurückgeben
    stats = {
        "trainer_runtime_seconds": trainer_runtime_seconds,
        "trainer_runtime_minutes": runtime_minutes,
        "start_reserved_memory_gb": start_memory,
        "peak_reserved_memory_gb": used_memory,
        "memory_for_training_gb": used_memory_for_lora,
        "peak_memory_percentage": used_percentage,
        "training_memory_percentage": lora_percentage,
    }

    return stats


# Funktion: Speicherstatistiken vor dem Training abrufen
def calculate_memory_before():
    """
    Gibt GPU-Informationen und initial reservierten Speicher zurück.
    :return: start_gpu_memory, max_memory, gpu_name
    """
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    gpu_name = gpu_stats.name

    print(f"GPU = {gpu_name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    return start_gpu_memory, max_memory, gpu_name

