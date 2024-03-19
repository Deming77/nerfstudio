import sqlite3
import csv
import copy
import os
from shutil import copyfile
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("input")
args = parser.parse_args()

# input_dir = "/hdd/data/aaviss-challenge1/t03_v01_s00_r03_ReconstructedArea"

# os.mkdir(os.path.join(input_dir,'images'))
# os.mkdir(os.path.join(input_dir,'poses'))

# os.system('mv '+input_dir+'*.png ', os.join.path(input_dir,'images'))
# os.system('mv '+input_dir+'*.json ', os.join.path(input_dir,'poses'))

# os.system('python wriva_to_colmap.py ' + input_dir)
colmap_dir = f"{args.input}/ta2_colmap/"

# os.system('colmap feature_extractor --database_path '+colmap_dir+'sparse/database.db --image_path images')
# os.system('colmap exhaustive_matcher --database_path '+colmap_dir+'sparse/database.db')

# os.mkdir(os.path.join(colmap_dir,'sparse/1'))
# os.mkdir(os.path.join(colmap_dir,'sparse/2'))
# os.mkdir(os.path.join(colmap_dir,'sparse/3'))
# os.system('colmap model_converter --input_path '+colmap_dir+'sparse/0 --output_path '+colmap_dir+'sparse/1 --output_type TXT')

connection = sqlite3.connect(colmap_dir+"sparse/0/database.db")
print("Connected to SQLite")
sql_query = """SELECT name FROM sqlite_master WHERE type='table';"""
cursor = connection.cursor()
cursor.execute(sql_query)
print("List of tables\n")
print(cursor.fetchall())

cursor.execute("select * from images;")
with open(colmap_dir+"sparse/2/images.csv", 'w',newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([i[0] for i in cursor.description])
    csv_writer.writerows(cursor)

print('wrote images.csv')
csv_list = []
txt_list = []
cursor.execute("select * from images;")
with open(colmap_dir+"sparse/2/images.csv", 'r') as csv_file:
    csv_reader = csv.reader(csv_file)
    for row in csv_reader:
        csv_list.append(row)
with open(colmap_dir+"sparse/1/images.txt", 'r') as read_txt:
    readlines = read_txt.readlines()
    for row in readlines:
        txt_list.append(row)

with open(colmap_dir+"sparse/2/images.txt", 'w') as write_txt:
    for row in txt_list:
        row_w = row
        if (row_w[0] == '#'):
            write_txt.write(row_w)
        if (row_w[0] != '#' and row_w != '\n'):
            name = row_w.split(" ")[-1].strip()
            for row2 in csv_list:
                if row2[1] == name:
                    new = row2[0] + " " + " ".join(row_w.split(" ")[1:])
            write_txt.write(new)
            write_txt.write('\n')

copyfile(colmap_dir+"sparse/1/cameras.txt", colmap_dir+"sparse/2/cameras.txt")
copyfile(colmap_dir+"sparse/1/points3D.txt", colmap_dir+"sparse/2/points3D.txt")

# os.system('colmap point_triangulator --database_path '+colmap_dir+'sparse/0/database.db --image_path '+input_dir+'images --input_path '+colmap_dir+'sparse/2 --output_path '+colmap_dir+'sparse/3')
