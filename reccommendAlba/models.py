from django.db import models


# Create your models here.
class OHaeng(models.Model):
    birth_date = models.CharField('Birth Date', max_length=10)
    tree = models.IntegerField('Tree', max_length=10)
    fire = models.IntegerField('Tree', max_length=10)
    soil = models.IntegerField('Tree', max_length=10)
    metal = models.IntegerField('Tree', max_length=10)
    water = models.IntegerField('Tree', max_length=10)
