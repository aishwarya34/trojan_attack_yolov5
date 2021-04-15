from PIL import Image
import glob
import os 
import csv 
import shutil
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


def main():

	array = []

	with open('confusion_matrix.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        array.append(row)

	
	df_cm = pd.DataFrame(array, index = [i for i in ["No Trojan", "Trojan"]],
	                  columns = [i for i in "01"])
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True)


if __name__ == "__main__":
    main()