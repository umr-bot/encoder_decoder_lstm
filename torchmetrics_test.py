# coding: utf-8
import torchmetrics
#preds = ["this is the prediction", "there is an other sample"]
#target = ["this is the reference", "there is another one"]
preds   = ["one","two"]
target  = ["oen","tow"]
metric = torchmetrics.CharErrorRate()
# float formatting: f'{value:{width}.{precision}}'
print(f"Char error rate: {metric(preds, target):.{2}}")

