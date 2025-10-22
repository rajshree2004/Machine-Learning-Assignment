import csv
import time

# ---------- Step 1: Read CSV ----------
filename = 'sales.csv'
data = []

with open(filename, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        # Clean column names and data
        row = {k.strip(): v.strip() for k, v in row.items()}
        row['Price'] = int(row['Price'])
        row['Quantity'] = int(row['Quantity'])
        data.append(row)

# ---------- Step 2: Store data in different structures ----------
list_data = [list(row.values()) for row in data]
tuple_data = [tuple(row.values()) for row in data]
set_data = {tuple(row.values()) for row in data}
dict_data = {int(row['ID']): row for row in data}

# ---------- Step 3: Retrieval Example ----------
def retrieve_by_product(structure, product):
    result = []
    if isinstance(structure, list):
        result = [item for item in structure if product in item]
    elif isinstance(structure, set):
        result = [item for item in structure if product in item]
    elif isinstance(structure, dict):
        result = [v for v in structure.values() if v['Product'] == product]
    elif isinstance(structure, tuple):
        result = [t for t in structure if product in t]
    return result

# ---------- Step 4: Filtering Example ----------
def filter_by_price(structure, min_price):
    if isinstance(structure, dict):
        return [v for v in structure.values() if v['Price'] >= min_price]
    else:
        result = []
        for item in structure:
            try:
                if int(item[2]) >= min_price:
                    result.append(item)
            except (ValueError, IndexError):
                continue
        return result

# ---------- Step 5: Aggregation Example ----------
def total_sales(structure):
    if isinstance(structure, dict):
        return sum(v['Price'] * v['Quantity'] for v in structure.values())
    else:
        total = 0
        for item in structure:
            try:
                total += int(item[2]) * int(item[3])
            except (ValueError, IndexError):
                continue
        return total

# ---------- Step 6: Compare Performance ----------
structures = {
    "List": list_data,
    "Tuple": tuple_data,
    "Set": set_data,
    "Dictionary": dict_data
}

product = input("Enter product name to search: ")
for name, struct in structures.items():
    start = time.time()
    retrieve_by_product(struct, product)
    filter_by_price(struct, 10)
    total_sales(struct)
    end = time.time()
    print(f"{name:10s} | Execution Time: {(end - start):.6f} seconds")
