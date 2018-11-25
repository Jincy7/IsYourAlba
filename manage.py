#!/usr/bin/env python
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import torch.optim as optim
import pandas as pd
from pandas import DataFrame, Series
import csv
from random import randrange


class myModel(nn.Module):
    def __init__(self):
        """
            In the constructor we instantiate two nn.Linear module
    """
        super(myModel, self).__init__()
        self.l1 = nn.Linear(13, 50)
        self.l2 = nn.Linear(50, 30)
        self.l3 = nn.Linear(30, 15)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
            In the forward function we accept a Variable of input data and we must return
            a Variable of output data. We can use Modules defined in the constructor as
            well as arbitrary operators on Variables.
    """
        x = x.view(-1, 13)
        out1 = self.sigmoid(self.l1(x))
        out2 = self.sigmoid(self.l2(out1))
        y_pred = self.sigmoid(self.l3(out2))
        return y_pred
if __name__ == '__main__':
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'IsYourAlba.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)
