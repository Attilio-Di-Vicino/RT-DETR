import csv

csv_file = 'Open Images V7 Extended Miap Boxes.csv' 
split = "validation"
txt_file = f'image_list_{split}.txt'
max_data = 5000

with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader) 
    
    with open(txt_file, 'w') as f:
        count = 0
        for row in reader:
            if count >= max_data: 
                break
            image_id = row[0]
            f.write(f'{split}/{image_id}\n')
            count += 1

print(f'File {txt_file} creato con successo con {count} righe!')