# Generated by Django 5.0.1 on 2024-04-27 20:36

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='informationstrees',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('humidity', models.TextField()),
                ('Tempreture', models.TextField()),
                ('step_count', models.TextField()),
            ],
        ),
    ]
