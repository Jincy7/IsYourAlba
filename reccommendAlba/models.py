from django.db import models


# Create your models here.
class OHaeng(models.Model):
    birth_date = models.CharField('Birth Date', max_length=10)
    tree = models.IntegerField('Tree', max_length=10)
    fire = models.IntegerField('Tree', max_length=10)
    soil = models.IntegerField('Tree', max_length=10)
    metal = models.IntegerField('Tree', max_length=10)
    water = models.IntegerField('Tree', max_length=10)


class Profile(models.Model):
    username = models.CharField('User Name', max_length=20)
    gender = models.CharField('User Gender', max_length=20, default="female")
    birth_date = models.CharField('Birth Date', max_length=10)
    type_EI = models.CharField('E or I', max_length=5)
    type_SN = models.CharField('S or N', max_length=5)
    type_TF = models.CharField('T or F', max_length=5)
    type_JP = models.CharField('J or P', max_length=5)
    num_I = models.IntegerField('I', max_length=5, default="0")
    num_E = models.IntegerField('E', max_length=5, default="0")
    num_S = models.IntegerField('S', max_length=5, default="0")
    num_N = models.IntegerField('N', max_length=5, default="0")
    num_T = models.IntegerField('T', max_length=5, default="0")
    num_F = models.IntegerField('F', max_length=5, default="0")
    num_J = models.IntegerField('J', max_length=5, default="0")
    num_P = models.IntegerField('P', max_length=5, default="0")


class Alba(models.Model):
    company_name = models.CharField('Company Name', max_length=100)
    gender_preference = models.CharField('Gender Preference', max_length=40)
    payment = models.CharField('Payment', max_length=40)
    work_type = models.CharField('Work Type', max_length=100)
    address = models.CharField('Address', max_length=100)
    url_link = models.CharField('Url Link', max_length=100)
