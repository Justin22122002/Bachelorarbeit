# Bachelor Thesis Projektstruktur

Diese README beschreibt die Struktur des Projekts für meine Bachelorarbeit. Die einzelnen Ordner und ihre Funktionen sind wie folgt:

## Verzeichnisübersicht

### `Bachelorthesis/`
Der Hauptordner des Projekts.

#### `DikeDataset/`
Dieser Ordner enthält das verarbeitete Datenset aus dem "DikesDataset". Hier werden Informationen wie die Anzahl der Samples und die Schadsoftwarefamilien extrahiert und in einer ZIP-Datei gespeichert. Die ZIP-Datei ist mit dem Passwort `infected` geschützt.

#### `LLM/`
In diesem Ordner wird das Training des Language Models (LLM) mit Unloth durchgeführt.

#### `malwareAnalysis/`
Dieser Ordner enthält alle Skripte und Tools, die für die Analyse von Schadsoftware-Samples notwendig sind.

- **`hwOutputParser/`**  
   Dieses Skript parst den Roh-Output, also alle JSON-Dateien, die von der Sandbox erzeugt werden. Es bereitet die Daten für die Analyse vor. Für die Analyse von pcap-Dateien gibt es nur Ansätze, jedoch keine vollständigen Lösungen.  
   Das Ergebnis dieses Skripts ist ein Datenset, das in weiteren Machine Learning-Anwendungen verwendet werden kann. Vor der Nutzung sollte das Skript `label_output_csv.py` im Verzeichnis `malwareAnalysis/malwareAnalysisTool/csv` ausgeführt werden, um die Daten zu labeln.
   Ebenfalls befinden sich hier Ansätze für eigenen Regeln die den Volatility Output nachanalysieren.

- **`malwareAnalysisTool/`**  
   Das Tool hier erstellt ein Datenset aus Speicherabbildern (Memory Dumps), das an das CIC Malmem 2022 Datenset angelehnt ist. Auch hier ist es notwendig, vor der Analyse das Skript `label_output_csv.py` auszuführen, um die Daten zu labeln.

- **`virustotal/`**  
   Dieser Ordner enthält ein Skript, das einen regelbasierten Ansatz verfolgt, um anhand des Filehashs über die VirusTotal API abzufragen, ob es sich bei einer Datei um Schadsoftware handelt oder nicht. Das Skript verwendet die API von VirusTotal, um die Datei zu überprüfen und die Ergebnisse zur weiteren Analyse bereitzustellen.

- **`volatility3/`**  
   **Wichtig:** Volatility3 muss vor der Nutzung des `malwareAnalysisTool` im Verzeichnis `malwareAnalysis/malwareAnalysisTool` geklont werden. Eine Anleitung zur Installation findest du in der `README`-Datei im Verzeichnis.

- **`yara/`**  
   Dieses Skript fasst alle YARA-Regeln aus einem GitHub-Repository zu einer Datei zusammen.

#### `malwareBazaar/`
Dieser Ordner enthält ein Tool, das es ermöglicht, Schadsoftware aus der MalwareBazaar API herunterzuladen. Es bietet verschiedene Abfragemöglichkeiten, wie nach Tags oder Signaturen.

#### `ml/`
In diesem Ordner befinden sich alle Dateien, die für den Machine Learning-Prozess verwendet werden, einschließlich Datensets und Skripte.

- **`datasets/`**  
   Hier werden alle Datensätze für das Training und die Analyse gespeichert.

## Weitere Hinweise
- Vor der Nutzung einiger Skripte müssen bestimmte Vorbereitungen getroffen werden, wie z.B. das Labeln von Ausgabedaten. Achte darauf, dass du alle notwendigen Schritte befolgst, um korrekte Ergebnisse zu erhalten.
- Weitere spezifische Anweisungen und Details sind in den jeweiligen Unterordnern oder in den README-Dateien innerhalb der Verzeichnisse zu finden.
