"""Python script that has functions to interpret and plot
   Tensorflow history object from model.fit outputs."""
import matplotlib.pyplot as plt
import pandas as pd

class Metric():
    """Initiate class object with input:
            fn: filename
            metric_names: metric types (in specific order) as a list
    """
    def __init__(self, fn, metric_names=["loss","val_loss","accuracy","val_accuracy","recall","f1_score","val_recall","val_f1_score"]):
        self.fn = fn
        self.metric_vals = []
        self.metric_names = metric_names
        self.df = pd.DataFrame(columns=self.metric_names)
        #self.decode()
    def read_fn(self):
        """Read metric values from history (returned from model.fit) file and
           store in self.metric_vals."""
        with open(self.fn) as f:
            for line in f:
                metric_vals = []
                for val in line.strip('\n').split(","): metric_vals.append(float(val))
                self.metric_vals.append(metric_vals)
    
    def decode(self):
        """Wrapper function to read in metric values from histry file."""
        self.read_fn() # load self.metric_vals with metric values
        return self.metric_vals

    def print_metrics(self):
        self.decode()
        for metric_val in self.metric_vals:
            for name,val in zip(self.metric_names, metric_val):
                print(f'{name}: {val:.4f}', end=" ")
            print("")
    def construct_dict(self):
        self.decode()
        for name,metric_val in zip(self.metric_names,self.metric_vals):
            self.df[name] = metric_val

    def plot_metrics(self):
        #metric.df[["accuracy","val_accuracy"]] makes a copy of df
        acc = metric.df.loc[:, ["accuracy","val_accuracy"]] # does not make a copy
        pass
#if "__name__" == "__main__":
metric = Metric(fn="checkpoints_eng_1/history_eng_1.txt")
#metric_vals = metric.decode()
metric.print_metrics()
