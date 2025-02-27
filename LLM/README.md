## Get Started

### Voraussetzungen

- Es wird empfohlen, **Anaconda** zu verwenden, da bestimmte Bibliotheken spezifische Python- und Library-Versionen
  benötigen.
- Eine **NVIDIA-Grafikkarte** wird benötigt, da dies die einzige kompatible Möglichkeit für das Training ist.

### Wichtige Versionen

- **Python:** 3.10
- **CUDA:** 12.1 (Neuere Versionen wie 12.4 sollten ebenfalls funktionieren, 12.1 wird jedoch empfohlen.)

### Setup mit Conda

1. Erstelle die Conda-Umgebung mit der Datei `requirements_conda.txt`:
   ```bash
   conda create --name <env> --file requirements_conda.txt 
   ```

2. Aktiviere die Umgebung:
    ```bash
    conda activate <env>
    ```
   
## Manuelles Setup (Falls die Conda-Datei nicht funktioniert)

### Conda-Umgebung erstellen
```conda create --name unsloth_env python=3.10 pytorch-cuda=12.4 pytorch cudatoolkit -c pytorch -c nvidia -c xformers -y```

### Conda-Umgebung aktivieren
```conda activate unsloth_env```

### Xformers installieren
Nur Version 0.0.27 funktioniert:
```pip install numpy```
```pip install xformers``` → ```pip install xformers==0.0.27```

### Die richtige PyTorch-Version installieren
Hinweis: Deinstalliere vorher torch, falls es bereits installiert ist, damit es mit cuda funktioniert:
```pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121```

### Unsloth installieren
```pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"``` oder, aber nicht empfohlen, ```pip install unsloth```
```pip install --no-deps trl peft accelerate bitsandbytes```

### Wenn du Windows verwendest, ist die einzige Möglichkeit, Triton zu installieren, eine spezifische .whl-Datei
Triton installieren → ```pip install https://github.com/woct0rdho/triton-windows/releases/download/v3.1.0-windows.post5/triton-3.1.0-cp310-cp310-win_amd64.whl```

### Modell von Hugging Face herunterladen
```huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --include "original/*" --local-dir Meta-Llama-3-70B-Instruct```
und verschiebe den Rest dieses Repositories in den Original-Ordner.
→ https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct/tree/main oder https://huggingface.co/unsloth/llama-3-8b-bnb-4bit/tree/main

**Hinweis**: Account bei Huggingface erstellen und Zugriff online anfragen !!! 


## Training starten

1. Stelle sicher, dass das Datenset im richtigen Ordner und im passenden Format vorliegt. Im folgenden Beispiel sollte das Format wie folgt aussehen:
    ```json
   {
        "instruction": "",
        "input": "",
        "output": ""
    } 

2. Führe das Notebook oder die Python-Skripte aus. Die Skripte sind für die spätere Weiterentwicklung vorgesehen, 
   während das Notebook für einen analytischen Ansatz genutzt wird.

3. Am Ende des Trainings werden Loara-Adapter und ein Snapshot des Trainings gespeichert. 
   Außerdem wird das Modell im GGUF-Format für Ollama exportiert:
   ```model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")```

4. Erstelle eine Modelfile, der festlegt, wie das Modell geladen wird. Diese Datei sollte folgendermaßen aussehen:
    ```
   FROM ./unsloth.F16.gguf
   
   SYSTEM du bist ein Schadsoftware Experte 
   der analytisch einem Nutzer Auskunft über das Verhalten einer Schadsoftware liefern soll, angand verschiedenener Parameter die als Input dienen
   ```
   Wobei **FROM** den Pfad zu dem gespeichertem GGUF-Format File festlegen soll
5. Starte das Modell mit dem folgenden Befehl:
   ``` ollama create <model_name> -f ./Modelfile```
   Name: Lama-Ware
   Das Modell ist inline verfügbar unter: https://huggingface.co/Justin22122002/Malware

6. Starte das Modell
   ``` ollama run <model_name> ```

