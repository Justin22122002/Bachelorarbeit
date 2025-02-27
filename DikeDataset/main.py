import csv
import os
import pyminizip
from DikeDataset.Parsing.CSVParser import CSVParser
from DikeDataset.Parsing.MalwareRecord import MalwareRecord


benign_files_path = "/media/justin/T7/DikeDataset-main/files/benign"
malware_files_path = "/media/justin/T7/DikeDataset-main/files/malware"

benign_files_path_output = "/media/justin/T7/samples/benign"
malware_files_path_output = "/media/justin/T7/samples/malware"

def load_benign_data(file_path: str) -> list[MalwareRecord]:
    """Lädt und gibt alle benignen Datensätze zurück."""
    parser = CSVParser(file_path)
    parser.load_data()
    records = parser.get_records()

    hashes = {record.hash for record in records}

    malware_records = [
        record for record in records
        if record.hash in hashes and 
        os.path.exists(os.path.join(benign_files_path, f"{record.hash}.exe"))
    ]

    return malware_records


def load_malware_data(file_path: str) -> list[MalwareRecord]:
    """Lädt und gibt alle Malware-Datensätze zurück."""
    parser = CSVParser(file_path)
    parser.load_data()
    records = parser.get_records()

    hashes = {record.hash for record in records}

    malware_records = [
        record for record in records
        if record.hash in hashes and 
        os.path.exists(os.path.join(malware_files_path, f"{record.hash}.exe"))
    ]

    return malware_records

def get_top_malware_samples(records: list[MalwareRecord], samples_per_type: int) -> dict[str, list[MalwareRecord]]:
    """Gibt ein Dictionary mit den höchsten Samples pro Malware-Typ zurück, vermeidet Duplikate und sortiert sie."""
    
    malware_classes = ["trojan", "ransomware", "worm", "backdoor", "spyware", "rootkit", "encrypter", "downloader"]
    
    top_samples_dict = {malware_class: [] for malware_class in malware_classes}

    for record in records:
        for malware_class in malware_classes:
            if hasattr(record, malware_class) and getattr(record, malware_class):
                top_samples_dict[malware_class].append(record)
    
    for malware_class in malware_classes:
        top_samples_dict[malware_class] = sorted(top_samples_dict[malware_class],
                                                   key=lambda x: getattr(x, malware_class), 
                                                   reverse=True)[:samples_per_type]

    return top_samples_dict

def count_duplicates(samples_dict: dict[str, list[MalwareRecord]]) -> int:
    seen_hashes = set()
    duplicates_count = 0
    
    for records in samples_dict.values():
        for record in records:
            record_hash = record.hash  # Verwende das hash-Attribut des MalwareRecord
            if record_hash in seen_hashes:
                duplicates_count += 1
            else:
                seen_hashes.add(record_hash)
    
    return duplicates_count


def save_samples(samples_dict: dict[str, list[MalwareRecord]], output_path: str, base_location: str):

    for malware_class, records in samples_dict.items():
        class_output_path = os.path.join(output_path, malware_class)
        os.makedirs(class_output_path, exist_ok=True)

        for record in records:
            file_path = os.path.join(base_location, f"{record.hash}.exe")
            
            if os.path.exists(file_path):
                zip_file_name = f"{record.hash}_{malware_class}.zip"
                zip_filepath = os.path.join(class_output_path, zip_file_name)

                try:
                    pyminizip.compress(file_path, None, zip_filepath, "infected", 0)
                    print(f"{os.path.basename(file_path)} zipped with password protection.")
                except Exception as e:
                    print(f"Error zipping {file_path}: {e}")
            else:
                print(f"File not found: {file_path}")


def save_to_csv(samples_dict: dict[str, list[MalwareRecord]], output_file: str):
    """Speichert die Malware-Proben in eine CSV-Datei."""
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Type', 'Hash', 'Malice', 'Generic', 'Trojan', 'Ransomware', 'Worm',
                         'Backdoor', 'Spyware', 'Rootkit', 'Encrypter', 'Downloader'])
        
        for malware_class, records in samples_dict.items():
            for record in records:
                writer.writerow([malware_class, record.hash, record.malice, record.generic, 
                                 record.trojan, record.ransomware, record.worm, 
                                 record.backdoor, record.spyware, record.rootkit, 
                                 record.encrypter, record.downloader])
                

def calculate_average_per_class(samples_dict: dict[str, list[MalwareRecord]]) -> dict[str, float]:
    """Berechnet den Durchschnittswert pro Malware-Klasse."""
    averages = {}
    
    for malware_class, records in samples_dict.items():
        if records:
            total_value = sum(getattr(record, malware_class) for record in records)
            averages[malware_class] = total_value / len(records)
        else:
            averages[malware_class] = 0.0
    
    return averages

def main():
    benign_file_path = './labels/benign.csv'
    malware_file_path = './labels/malware.csv'

    benign_records = load_benign_data(benign_file_path)

    malware_records = load_malware_data(malware_file_path)

    samples_per_type = 100

    top_malware_samples_dict = get_top_malware_samples(malware_records, samples_per_type)

    print("Top Malware Samples by Class:")
    for malware_class, records in top_malware_samples_dict.items():
        print(f"\n{malware_class}: {len(records)} samples")

    duplicate_count = count_duplicates(top_malware_samples_dict)
    print(f"\nTotal duplicates found: {duplicate_count}")

    save_samples(top_malware_samples_dict, malware_files_path_output, malware_files_path)


    reduced_samples = {"benign" : benign_records[:300]}
    save_samples(reduced_samples, benign_files_path_output, benign_files_path)

    save_to_csv(top_malware_samples_dict, '/media/justin/T7/samples/top_malware_samples.csv')
    save_to_csv(reduced_samples, '/media/justin/T7/samples/benign_samples.csv')
    
    averages = calculate_average_per_class(top_malware_samples_dict)
    print("\nAverage values per malware class:")
    for malware_class, avg in averages.items():
        print(f"{malware_class}: {avg}")

if __name__ == "__main__":
    main()
