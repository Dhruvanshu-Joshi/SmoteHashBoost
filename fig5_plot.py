import pandas as pd
from matplotlib import pyplot as plt
# from google.colab import files
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

# 0.247, 0.747
# 0.9926
# 0.9274
# 0.9333
# 0.8718
# 0.5133 ,0.9788, 0.9818, 0.9361, 0.9682, 0.7928, 0.853, 0.426,0.4415

# {'t_AB': '', 't_AS': '0.2158,0.7554,0.9926,0.8871,0.2609,0.7477,0.984,0.9098,0.2609,0.7477,0.984,0.9098,0.8718,0.4961,0.9644,0.9682,0.9265,0.97,0.7737,0.3816,0.4072', '_BLS': '', 't_HB': '', '_HUE': '', 't_RB': '0.2761,0.6649,0.8994,0.8197,0.9,0.9138,0.5083,0.9788,0.9905,0.9617,0.9823,0.8032,0.2761,0.6649,0.8994,0.8197,0.9,0.9138,0.5083,0.9788,0.9905,0.9617,0.9823,0.8032,0.2761,0.6649,0.8994,0.8197,0.9,0.9138,0.5083,0.9788,0.9905,0.9617,0.9823,0.8032,0.2761,0.6649,0.8994,0.8197,0.9,0.9138,0.5083,0.9788,0.9905,0.9617,0.9823,0.8032,0.2761,0.6649,0.8994,0.8197,0.9,0.9138,0.5083,0.9788,0.9905,0.9617,0.9823,0.8032', 't_SB': '', 'SENN': '0.3532,0.7548,0.9623,0.836,0.9333,0.8775,0.5738,0.9783,0.9462,0.9196,0.9282,0.7315,0.8088,0.416,0.4525', '_SHB': '0.247,0.747,0.9926,0.9274,0.9333,0.8718,0.5133,0.9788,0.9818,0.9361,0.9682,0.7928,0.853,0.426,0.4415', 't_ST': ''}
# {'t_AB': [''], 't_AS': ['Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Epileptic Seizure Recognition', 'Student_dropout', 'default of credit card clients'], '_BLS': ['Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Epileptic Seizure Recognition', 'Student_dropout', 'default of credit card clients'], 't_HB': ['Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Epileptic Seizure Recognition', 'Student_dropout', 'default of credit card clients'], '_HUE': ['Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Epileptic Seizure Recognition', 'Student_dropout', 'default of credit card clients'], 't_RB': ['Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar'], 't_SB': ['Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Epileptic Seizure Recognition', 'Student_dropout', 'default of credit card clients', 'Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Epileptic Seizure Recognition', 'Student_dropout', 'default of credit card clients'], 'SENN': ['Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Epileptic Seizure Recognition', 'Student_dropout', 'default of credit card clients'], '_SHB': ['Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Epileptic Seizure Recognition', 'Student_dropout', 'default of credit card clients'], 't_ST': ['Flare-F', 'Yeast5', 'CarvGood', 'CarGood', 'Yeast5-ERL', 'Glass', 'ILPD', 'Seed', 'Wine', 'Breast Cancer Wisconsin', 'Diabetes', 'Sonar', 'Epileptic Seizure Recognition', 'Student_dropout', 'default of credit card clients']}

# 0.283,0.5943,0.8314,0.6456,0.6267,0.8247,0.5774,0.9788,0.961,

#f1_gmean
plotdata = pd.DataFrame({
    "SMOTEHashBoost":[0.3036, 0.815, 0.9926, 0.9316, 0.9333, 0.9905, 0.9857, 0.8816, 0.5244, 0.9361, 0.9682, 0.7928, 0.853, 0.426,0.4415],
    "RUSBoost":[0.2842, 0.6691, 0.8755, 0.8424, 0.8667, 0.9895, 0.9788, 0.9113, 0.5275, 0.9617,0.9823,0.8032, 0, 0, 0],
    "HUE":[0.2771, 0.586, 0.829, 0.6371, 0.6467, 0.9695, 0.9788, 0.8242, 0.5825, 0.9359,0.9659,0.7977,0.87,0.4939,0.4896],
    "HashBoost": [0.2796,0.5997,0.8318,0.6486,0.6364,0.8275,0.5768,0.9714,0.9714,0.9399,0.963,0.7961,0.8713,0.4903,0.4906], 
    "SMOTEBoost":[0.2334,0.7248,0.992,0.89,0.9333,0.8597,0.5033,0.9777,0.9701,0.9254,0.9655,0.7314,0.8028,0.3809,0.4047],
    "AdaBoost": [0.1851,0.7167,0.9846,0.9926,0.9333,0.8667,0.487,0.9028,0.9789,0.9252,0.9651,0.7394,0.8547,0.3795,0.4102],
    "SMOTE-ENN": [0.3532,0.7548,0.9623,0.836,0.9333,0.8775,0.5738,0.9783,0.9462,0.9196,0.9282,0.7315,0.8088,0.416,0.4525],
    "SMOTE-Tomek": [0.272,0.7591,0.9846,0.8799,0.9333,0.8818,0.5051,0.9709,0.9704,0.9317,0.968,0.7393,0.8102,0.3858,0.4058],
    "Borderline-SMOTE": [0.2554,0.7107,0.9926,0.8446,0.7333,0.8779,0.4851,0.9575,0.9789,0.9158,0.9697,0.7661,0.8155,0.3837,0.4069],
    "ADASYN": [0.2609,0.7477,0.984,0.9098,0,0.8718,0.4961,0.9644,0.9682,0.9265,0.97,0,0.7737,0.3816,0.4072],
    "GAN": [0.9558, 0.9677, 0.9478, 0.9425, 0.0, 0.5674, 0.6752, 0.8714, 0.7389, 0.8978, 0.9197, 0.8679, 0.9589, 0.6380, 0.0],
    "SMOTified-GAN": [0.9593, 0.9623, 0.9314, 0.9309, 0.0, 0.5547, 0.7009, 0.8631, 0.7667, 0.9417, 0.9298, 0.8762, 0.9584, 0.6446, 0.0]
    },
    index=['Flare-F', 'Yeast5', 'Carvgood', 'Cargood', 'Yeast5-ERL', 'Wine', 'Seed', 'Glass', 'ILPD', 'ESDRP ', 'CB', 'BCW', 'DCCC', 'ESR', 'PSDAS']
)
# plotdata.plot(kind="bar", linewidth=5)
# #plt.title("Mince Pie Consumption Study")
# #plt.ylabel('SCH (ms)', fontsize = 15, style = 'italic')
# plt.xlabel('Dataset', fontsize = 15, style = 'italic')
# plt.legend(loc = 4,prop={'size': 15}, edgecolor = 'black', frameon = True, framealpha=0.5)
# #plt.title("SCH", fontsize = 15)
# #plt.ylim([50, 100])
# plt.savefig("F1.pdf", bbox_inches='tight')
# files.download("F1.pdf")
#plt.show()


# import pandas as pd
# from matplotlib import pyplot as plt
# # from google.colab import files
# import matplotlib

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rc('xtick', labelsize=15)
# matplotlib.rc('ytick', labelsize=15)

# plotdata = pd.DataFrame({
#     "SMOTEHashBoost":[0.9818, 0.1315, 0.6837, 0.9857, 0.9668, 0.9772, 0.8136, 0.397, 0.9, 0.9004,0.9484,0.7314,0.7451,0.2855,0.3233],
#     "RUSBoost":[0.9856, 0.1493, 0.4954, 0.7842, 0.7289, 0.9677, 0.8576, 0.4107, 0.8, 0.9433,0.9749,0.7527,0.0,0.0,0.0],
#     "HUE":[0.1575,0.4354,0.7153,0.4827,0.54,0.7124,0.4239,0.9639,0.9273,0.8903,0.9396,0.7258,0.7705,0.322,0.33],
#     "HashBoost": [0.1546,0.4303,0.7209,0.4802,0.5533,0.7102,0.4179,0.9544,0.9455,0.9007,0.9375,0.713,0.7727,0.3188,0.3306], 
#     "SMOTEBoost":[0.1096,0.5422,0.9852,0.801,0.9,0.7919,0.3882,0.9676,0.9502,0.8815,0.9426,0.6673,0.6764,0.2533,0.2915],
#     "AdaBoost": [0.0811,0.5381,0.9709,0.9857,0.9,0.8047,0.3843,0.8596,0.96,0.8812,0.9453,0.6593,0.7606,0.2537,0.2955],
#     "SMOTEEnn": [0.1754,0.605,0.9295,0.7217,0.9,0.7997,0.4256,0.9676,0.9171,0.8786,0.8772,0.6762,0.6842,0.2656,0.3103],
#     "SMOTETomek": [0.1145,0.5924,0.9709,0.7837,0.9,0.813,0.3915,0.9581,0.9436,0.8957,0.9454,0.673,0.6869,0.257,0.2925],
#     "BorderlineSMOTE": [0.1055,0.5319,0.9857,0.7287,0.7,0.8005,0.3763,0.9331,0.9711,0.8699,0.9528,0.7092,0.6941,0.252,0.2925],
#     "ADASYN": [0.1082,0.5742,0.9704,0.8372,0.0,0.7822,0.3807,0.9464,0.9509,0.8787,0.9528,0.0,0.6323,0.2536,0.2926],
    # "GAN": [0.4368, 0.4603, 0.9876, 0.9882, 0.0, 0.6489, 0.4599, 0.9618, 0.8572, 0.9892, 0.9942, 0.9363, 0.9460, 0.6887, 0.0], 
    # "SMOTified-GAN": [0.3954, 0.4473, 0.9838, 0.9773, 0.0, 0.6485, 0.4775, 0.9621, 0.8839, 0.9931, 0.9955, 0.9328, 0.9468, 0.7016, 0.0]
#     },
#     index=['Flare-F', 'Yeast5', 'Carvgood', 'Cargood', 'Yeast5-ERL', 'Wine', 'Seed', 'Glass', 'ILPD', 'ESDRP ', 'CB', 'BCW', 'DCCC', 'ESR', 'PSDAS']
# )
# plotdata.plot(kind="bar", linewidth=0)
# #plt.title("Mince Pie Consumption Study")
# #plt.ylabel('SCH (ms)', fontsize = 15, style = 'italic')
# plt.xlabel('Dataset', fontsize = 15, style = 'italic')
# plt.legend(loc = 4,prop={'size': 15}, edgecolor = 'black', frameon = True, framealpha=0.5)
# #plt.title("SCH", fontsize = 15)
# #plt.ylim([50, 100])
# plt.savefig("AP.pdf", bbox_inches='tight')
# # files.download("AP.pdf")
# #plt.show()


# import pandas as pd
# from matplotlib import pyplot as plt
# import matplotlib

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.rc('xtick', labelsize=15)
# matplotlib.rc('ytick', labelsize=15)

# plotdata = pd.DataFrame({
    # "SMOTEHashBoost":[0.9961, 0.5578, 0.9485, 0.9997, 0.968, 0.9893, 0.9441, 0.6574, 0.9997, 0.9495,0.9758,0.7994,0.9427,0.612,0.6027],
    # "RUSBoost":[0.9922, 0.8353, 0.9722, 0.9943, 0.9921, 0.9856, 0.9407, 0.6568, 0.9993, 0.9675,0.9842,0.8186,0.0,0.0,0.0],
    # "HUE":[0.9819, 0.8558, 0.9666, 0.9915, 0.9756, 0.9856, 0.9268, 0.6997, 0.997, 0.9536,0.9751,0.8073,0.9562,0.7411,0.6851],
    # "HashBoost": [0.8555,0.9677,0.9918,0.9771,0.997,0.936,0.6947,0.9784,0.9883,0.9549,0.972,0.793,0.9556,0.737,0.6861], 
    # "SMOTEBoost":[00.4335,0.8738,0.9922,0.9367,0.9997,0.9153,0.639,0.9816,0.9807,0.9415,0.9733,0.7468,0.8915,0.5842,0.5857],
    # "AdaBoost": [0.3675,0.8463,0.9919,0.9997,0.9997,0.939,0.6225,0.9228,0.9922,0.9417,0.9715,0.7478,0.9057,0.5794,0.5892],
    # "SMOTE-ENN": [0.6522,0.9683,0.9834,0.9489,0.9997,0.9344,0.6978,0.9818,0.9603,0.9344,0.9461,0.7532,0.8973,0.6437,0.6431],
    # "SMOTE-Tomek": [0.4906,0.8974,0.9919,0.9438,0.9997,0.9215,0.6374,0.9746,0.9883,0.9444,0.9759,0.7509,0.8958,0.5896,0.5864],
    # "Borderline-SMOTE": [0.4687,0.8958,0.9997,0.9355,0.7997,0.9352,0.6237,0.9673,0.9795,0.9331,0.9754,0.7837,0.901,0.5939,0.5882],
    # "ADASYN": [0.5064,0.8829,0.9843,0.9392,0.0,0.935,0.6333,0.9709,0.9744,0.9446,0.9756,0.0,0.8882,0.584,0.5886],
    # "GAN": [0.5297, 0.6557, 0.9320, 0.8987, 0.0, 0.0000, 0.3687, 0.8576, 0.5401, 0.8965, 0.9126, 0.8618, 0.9229, 0.3172, 0.0],
    # "SMOTified-GAN": [0.4387, 0.5501, 0.8185, 0.8488, 0.0, 0.0000, 0.4350, 0.8524, 0.5495, 0.9365, 0.9196, 0.8775, 0.9223, 0.3221, 0.0]
    # },
#     index=['Flare-F', 'Yeast5', 'Carvgood', 'Cargood', 'Yeast5-ERL', 'Wine', 'Seed', 'Glass', 'ILPD', 'ESDRP ', 'CB', 'BCW', 'DCCC', 'ESR', 'PSDAS']
# )

# Define unique color palette
unique_colors = plt.cm.get_cmap("tab20", len(plotdata.columns)).colors

fig, ax = plt.subplots(figsize=(20, 8))  # Wider figure for better spacing
plotdata.plot(kind="bar", width=0.8, ax=ax, color=unique_colors)  # Adjust bar width for thickness


plt.xlabel('Dataset', fontsize = 15, style = 'italic')
plt.legend(loc = 4,prop={'size': 15}, edgecolor = 'black', frameon = True, framealpha=0.5)

ax.get_legend().remove()

plt.savefig("GM_4.pdf", bbox_inches='tight')

# plt.show()

# Second plot (only legend)
fig_leg = plt.figure(figsize=(15, 1))  # Adjust figure size for horizontal layout
ax_leg = fig_leg.add_subplot(111)
ax_leg.axis("off")  # Hide axis

handles, labels = ax.get_legend_handles_labels()
legend = ax_leg.legend(handles, labels, loc="center", ncol=len(labels) // 2, prop={'size': 12}, edgecolor='black', frameon=True, framealpha=0.5)

plt.savefig("Legend_GM.pdf", bbox_inches='tight')
# plt.show()

# # Plot with horizontal legend (Second PDF)
# fig, ax = plt.subplots(figsize=(20, 8))
# plotdata.plot(kind="bar", width=0.8, ax=ax, color=unique_colors)

# plt.xlabel('Dataset', fontsize=15, style='italic')

# # Add horizontal legend
# legend = plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
#                     ncol=len(plotdata.columns)//2, fontsize=12, edgecolor='black')

# plt.savefig("F1_4_legend.pdf", bbox_inches='tight')
# plt.show()
