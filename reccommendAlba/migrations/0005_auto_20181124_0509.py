# Generated by Django 2.1.3 on 2018-11-23 20:09

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('reccommendAlba', '0004_auto_20181124_0426'),
    ]

    operations = [
        migrations.AddField(
            model_name='profile',
            name='num_E',
            field=models.IntegerField(default='0', max_length=5, verbose_name='E'),
        ),
        migrations.AddField(
            model_name='profile',
            name='num_F',
            field=models.IntegerField(default='0', max_length=5, verbose_name='F'),
        ),
        migrations.AddField(
            model_name='profile',
            name='num_I',
            field=models.IntegerField(default='0', max_length=5, verbose_name='I'),
        ),
        migrations.AddField(
            model_name='profile',
            name='num_J',
            field=models.IntegerField(default='0', max_length=5, verbose_name='J'),
        ),
        migrations.AddField(
            model_name='profile',
            name='num_N',
            field=models.IntegerField(default='0', max_length=5, verbose_name='N'),
        ),
        migrations.AddField(
            model_name='profile',
            name='num_P',
            field=models.IntegerField(default='0', max_length=5, verbose_name='P'),
        ),
        migrations.AddField(
            model_name='profile',
            name='num_S',
            field=models.IntegerField(default='0', max_length=5, verbose_name='S'),
        ),
        migrations.AddField(
            model_name='profile',
            name='num_T',
            field=models.IntegerField(default='0', max_length=5, verbose_name='T'),
        ),
    ]
