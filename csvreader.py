csv_path = "./albamon.csv"
django_home = "C:\\Users\\진창엽\\PycharmProjects\\IsYourAlba"

import os, sys
sys.path.append(django_home)
os.environ['DJANGO_SETTINGS_MODULE'] = 'IsYourAlba.settings'

from reccommendAlba.models import Alba

import csv

dataReader = csv.reader(open(csv_path), delimiter=',')
alba = None
for row in dataReader:
    alba = Alba()
    alba.company_name = row[0]
    alba.gender_preference = row[1]
    alba.payment = row[2]
    alba.work_type = row[3]
    alba.address = row[4]
    alba.url_link = row[5]

    alba.save()
