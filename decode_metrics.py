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
        # call initialization functions
        self.read_fn()
        self.construct_dataframe()

    def read_fn(self):
        """Read metric values from history (returned from model.fit) file and
           store in self.metric_vals."""
        with open(self.fn) as f:
            for line in f:
                metric_vals = []
                for val in line.strip('\n').split(","): metric_vals.append(float(val))
                self.metric_vals.append(metric_vals)
    
    def print_metrics(self,metric_names=["accuracy","val_accuracy"]):
        #for metric_val in self.metric_vals:
            #for name,val in zip(self.metric_names, metric_val): print(f'{name}: {val:.4f}', end=" ")
            #print("")
        # print all rows in columns specified with metric_names
        #metric.df[["accuracy","val_accuracy"]] makes a copy of df
        print(self.df.loc[:, metric_names]) # not a copy of df
   
    def construct_dataframe(self):
        """Create pandas dataframe out of metric values and columns extracted
           from tensorflow model.fit history object"""
        self.df = pd.DataFrame(self.metric_vals,columns=list(self.df.columns))

    def plot_metrics(self,metric_names=["accuracy","val_accuracy"]):
        x_range = list(range(self.df.shape[0])) #range with number of rows in df
        for i in range(len(metric_names)):
            # assign to y only the values and not indices out of df object
            plt.plot(x_range, list(self.df[metric_names[i]]))
        #plt.plot(x, self.df[metric_names[1], '-.')

        plt.xlabel("Epochs")
        plt.ylabel("Metric scores")
        plt.title('multiple plots')
        plt.show()

#if "__name__" == "__main__":
metric = Metric(fn="checkpoints_eng_3/history_eng_3.txt")
#metric = Metric(fn="checkpoints_bam_1/history_bam_1.txt")

#metric_vals = metric.decode()
#metric.print_metrics()
