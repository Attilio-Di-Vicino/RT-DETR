import csv

csv_file = 'Open Images V7 Extended Miap Boxes.csv' 
split = "validation"
txt_file = f'image_list_{split}.txt'

with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    
    with open(txt_file, 'w') as f:
        for row in reader:
            image_id = row[0]
            f.write(f'{split}/{image_id}\n')

print(f'File {txt_file} creato con successo!')
