data = [{'sales': 200}, {'sales': 220}, {'sales': 210}, {'sales': 250}, {'sales': 300}]

for i in range(len(data)):
    data[i]['sales_lag_1'] = data[i - 1]['sales'] if i >= 1 else None
    data[i]['sales_lag_3'] = data[i - 3]['sales'] if i >= 3 else None

print(data