import csv

csv_file = 'Open Images V7 Extended Miap Boxes Test.csv'  
txt_file = 'image_list.txt'

with open(csv_file, 'r') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    
    with open(txt_file, 'w') as f:
        for row in reader:
            image_id = row[0]
            f.write(f'test/{image_id}\n')

print(f'File {txt_file} creato con successo!')
