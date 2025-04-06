import pandas as pd
from matplotlib import pyplot as plt
from google.colab import files
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

plotdata = pd.DataFrame({
    "SMOTEHashBoost":[0.9905, 0.3036, 0.815, 0.9926, 0.9316, 0.9857, 0.8816, 0.5244, 0.9333],
    "RUSBoost":[0.9895, 0.2842, 0.6691, 0.8755, 0.8424, 0.9788, 0.9113, 0.5275, 0.8667],
    "HUE":[0.9695, 0.2771, 0.586, 0.829, 0.6371, 0.9788, 0.8242, 0.5825, 0.6467]
    },
    index=["Flare-F", "Yeast5", "CarvGood", "CarGood", "Glass", "ILPD", "Seed", "Wine", "Breast Cancer Wisconsin", "Diabetes", "Epileptic Seizure Recognition", "Student_dropout", "default of credit card clients"]
)
plotdata.plot(kind="bar", linewidth=0)
#plt.title("Mince Pie Consumption Study")
#plt.ylabel('SCH (ms)', fontsize = 15, style = 'italic')
plt.xlabel('Dataset', fontsize = 15, style = 'italic')
plt.legend(loc = 4,prop={'size': 15}, edgecolor = 'black', frameon = True, framealpha=0.5)
#plt.title("SCH", fontsize = 15)
#plt.ylim([50, 100])
plt.savefig("F1.pdf", bbox_inches='tight')
files.download("F1.pdf")
#plt.show()