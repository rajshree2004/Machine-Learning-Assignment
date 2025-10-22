import csv
import time

# ---------- Step 1: Read CSV ----------
filename = "data.csv"
data_list = []

with open(filename, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        row['Salary'] = int(row['Salary'])  # Convert salary to integer
        data_list.append(row)

print("Data loaded successfully!\n")


# ---------- Step 2: Store in Different Data Structures ----------

# List of dictionaries
data_list_dict = data_list.copy()

# Tuple of tuples
data_tuple = tuple(tuple(row.values()) for row in data_list)

# Set of tuples (unique rows only)
data_set = set(data_tuple)

# Dictionary (keyed by ID)
data_dict = {int(row['ID']): row for row in data_list}


# ---------- Step 3: Define Operations ----------

def filter_by_department(data, dept):
    """Return all records from a specific department."""
    if isinstance(data, list):
        return [d for d in data if d['Department'] == dept]
    elif isinstance(data, tuple):
        return [d for d in data if d[2] == dept]
    elif isinstance(data, set):
        return [d for d in data if d[2] == dept]
    elif isinstance(data, dict):
        return [v for v in data.values() if v['Department'] == dept]

def aggregate_salary(data):
    """Return total and average salary."""
    if isinstance(data, list):
        salaries = [d['Salary'] for d in data]
    elif isinstance(data, tuple):
        salaries = [d[3] for d in data]
    elif isinstance(data, set):
        salaries = [d[3] for d in data]
    elif isinstance(data, dict):
        salaries = [v['Salary'] for v in data.values()]
    
    total = sum(salaries)
    avg = total / len(salaries)
    return total, avg

def retrieve_by_id(data, emp_id):
    """Retrieve employee by ID."""
    if isinstance(data, list):
        return next((d for d in data if int(d['ID']) == emp_id), None)
    elif isinstance(data, tuple):
        return next((d for d in data if int(d[0]) == emp_id), None)
    elif isinstance(data, set):
        return next((d for d in data if int(d[0]) == emp_id), None)
    elif isinstance(data, dict):
        return data.get(emp_id, None)


# ---------- Step 4: Performance Comparison ----------

structures = {
    "List": data_list_dict,
    "Tuple": data_tuple,
    "Set": data_set,
    "Dictionary": data_dict
}

print("Performance Comparison:\n")

for name, dataset in structures.items():
    print(f"--- {name} ---")
    
    start = time.time()
    filter_result = filter_by_department(dataset, "Engineering")
    end = time.time()
    print(f"Filtering Time: {end - start:.6f} sec")

    start = time.time()
    agg_total, agg_avg = aggregate_salary(dataset)
    end = time.time()
    print(f"Aggregation Time: {end - start:.6f} sec")

    start = time.time()
    emp = retrieve_by_id(dataset, 3)
    end = time.time()
    print(f"Retrieval Time: {end - start:.6f} sec\n")


# ---------- Step 5: Display Example Results ----------
print("Example Results:\n")
print("Filter (Engineering Dept):", filter_by_department(data_list_dict, "Engineering"))
print("Aggregate Salary:", aggregate_salary(data_list_dict))
print("Retrieve ID=3:", retrieve_by_id(data_list_dict, 3))
