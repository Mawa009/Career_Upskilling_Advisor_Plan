
import csv

file_path = r"D:\AIcon26\skills_en.csv"

with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    print("Column names:")
    print(reader.fieldnames)

    print("\nFirst 5 rows:")
    for i, row in enumerate(reader):
        print(row)
        if i == 4:
            break
