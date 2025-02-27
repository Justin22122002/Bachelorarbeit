from datasets import load_dataset
from unsloth import FastLanguageModel

from LLM.trainer.LanguageModelHandler import LanguageModelHandler
from LLM.trainer.TrainerHandler import TrainerHandler
from LLM.trainer.helper import calculate_memory_before, calculate_memory_after, formatting_prompts_func
from transformers import TextStreamer


def run_prompt(model, tokenizer, alpaca_prompt, instruction, _input):
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction,  # instruction
                _input,  # input
                "",  # output
            )
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    tokenizer.batch_decode(outputs)


def run_prompt_stream(model, tokenizer, alpaca_prompt, instruction, _input):
    FastLanguageModel.for_inference(model)
    inputs = tokenizer(
        [
            alpaca_prompt.format(
                instruction,  # instruction
                _input,  # input
                "",  # output
            )
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=512, use_cache=True)
    tokenizer.batch_decode(outputs)

    text_streamer = TextStreamer(tokenizer)

    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=512,
        use_cache=True,
    )


def main() -> None:
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    model_handler = LanguageModelHandler(
        model_path="D:/Users/Groh_Justin/PycharmProjects/LLM/models/Meta-Llama-3-70B-Instruct/original"
    )

    # Load the model
    model_handler.load_model()

    # Apply PEFT
    model_handler.apply_peft()

    dataset_path = "dataset_Malware/malware_analysis_dataset_2.json"
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    dataset = dataset.map(
        lambda examples: formatting_prompts_func(examples, model_handler.tokenizer, alpaca_prompt),
        batched=True,
    )

    # Initialize the Trainer Handler
    trainer_handler = TrainerHandler(
        model=model_handler.model,
        tokenizer=model_handler.tokenizer,
        dataset=dataset,
        max_seq_length=model_handler.max_seq_length,
    )

    trainer_handler.create_trainer()

    start_memory, max_memory, gpu_name = calculate_memory_before()

    print("Starte das Training...")
    trainer_stats = trainer_handler.train_model()
    trainer_runtime_seconds = trainer_stats.metrics['train_runtime']  # Beispielwert für 1 Stunde Training

    memory_stats = calculate_memory_after(trainer_runtime_seconds, start_memory, max_memory)

    # Ausgabe der Ergebnisse nach dem Training
    print(f"\nTraining beendet.")
    print(f"Training dauerte {memory_stats['trainer_runtime_seconds']} Sekunden.")
    print(f"Das entspricht {memory_stats['trainer_runtime_minutes']} Minuten.")
    print(f"Zu Beginn reservierter Speicher: {memory_stats['start_reserved_memory_gb']} GB.")
    print(f"Peak reservierter Speicher: {memory_stats['peak_reserved_memory_gb']} GB.")
    print(f"Speicher für das Training genutzt: {memory_stats['memory_for_training_gb']} GB.")
    print(f"Peak reservierter Speicher (Prozent von Max): {memory_stats['peak_memory_percentage']} %.")
    print(f"Speicher für Training (Prozent von Max): {memory_stats['training_memory_percentage']} %.")

    instruction = "Analyze metrics to detect potential malware activities."
    _input = "pslist_nproc: 142, pslist_nppid: 20, pslist_avg_threads: 13.147887323943662, pslist_nprocs64bit: 142, pslist_avg_handlers: 0.0, dlllist_ndlls: 7190, dlllist_avg_dlls_per_proc: 52.48175182481752, handles_nhandles: 50211, handles_avg_handles_per_proc: 363.8478260869565, handles_nport: 0, handles_nfile: 2629, handles_nevent: 11455, handles_ndesktop: 145, handles_nkey: 3754, handles_nthread: 2252, handles_ndirectory: 322, handles_nsemaphore: 2614, handles_ntimer: 341, handles_nsection: 972, handles_nmutant: 837, ldrmodules_not_in_load: 401, ldrmodules_not_in_init: 540, ldrmodules_not_in_mem: 401, ldrmodules_not_in_load_avg: 0.053042328042328, ldrmodules_not_in_init_avg: 0.0714285714285714, ldrmodules_not_in_mem_avg: 0.053042328042328, malfind_ninjections: 24, malfind_commitCharge: 1889, malfind_protection: 24, malfind_uniqueInjections: 6, psxview_not_in_pslist: 3, psxview_not_in_eprocess_pool: 0, psxview_not_in_ethread_pool: 4, psxview_not_in_pspcid_list: 145, psxview_not_in_csrss_handles: 10, psxview_not_in_session: 0, psxview_not_in_deskthrd: 145, psxview_not_in_pslist_false_avg: 0.0206896551724137, psxview_not_in_eprocess_pool_false_avg: 0.0, psxview_not_in_ethread_pool_false_avg: 0.0275862068965517, psxview_not_in_pspcid_list_false_avg: 1.0, psxview_not_in_csrss_handles_false_avg: 0.0689655172413793, psxview_not_in_session_false_avg: 0.0, psxview_not_in_deskthrd_false_avg: 1.0, modules_nmodules: 175, svcscan_nservices: 1473, svcscan_kernel_drivers: 750, svcscan_fs_drivers: 85, svcscan_process_services: 151, svcscan_shared_process_services: 2, svcscan_interactive_process_services: 2, svcscan_nactive: 402, callbacks_ncallbacks: 258, callbacks_nanonymous: 217, callbacks_ngeneric: 19",

    run_prompt(model_handler.model, model_handler.tokenizer, alpaca_prompt, instruction, _input)
    run_prompt_stream(model_handler.model, model_handler.tokenizer, alpaca_prompt, instruction, _input)


if __name__ == "__main__":
    main()
