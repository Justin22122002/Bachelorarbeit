from DikeDataset.Parsing.MalwareRecord import MalwareRecord

import csv


class CSVParser:
    def __init__(self, file_path: str) -> None:
        self.file_path: str = file_path
        self.records: list[MalwareRecord] = []

    def load_data(self) -> None:
        with open(self.file_path, mode='r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                record = MalwareRecord(
                    type=row['type'],
                    hash=row['hash'],
                    malice=float(row['malice']),
                    generic=float(row['generic']),
                    trojan=float(row['trojan']),
                    ransomware=float(row['ransomware']),
                    worm=float(row['worm']),
                    backdoor=float(row['backdoor']),
                    spyware=float(row['spyware']),
                    rootkit=float(row['rootkit']),
                    encrypter=float(row['encrypter']),
                    downloader=float(row['downloader'])
                )
                self.records.append(record)

    def get_records(self) -> list[MalwareRecord]:
        return self.records
