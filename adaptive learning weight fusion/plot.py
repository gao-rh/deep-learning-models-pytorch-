# %%
import matplotlib.pyplot as plt
import pandas as pd

#read data
df = pd.read_excel('./train_info/weights_plot.xlsx')
list=[df]
for i in range(5):
    list.append(df.groupby(2).get_group(i))
list[0][0]

#plot
plt.rcParams['font.sans-serif'] = 'times new roman'
plt.rcParams['savefig.dpi'] = 300
list_name = ['all','residential', 'commercial zones', 'education ', 'green land', 'public service']
fig, axs = plt.subplots(3,2,constrained_layout=True)
kwargs =dict(bins=20, density=True, alpha=0.9, histtype='stepfilled', color='steelblue', edgecolor='none')
for i in range(3):
    for j in range(2):
        ax = axs[i,j]
        ax.hist(list[j+i*2][0],**kwargs)
        ax.axis(xmin=0, xmax=1)
        ax.set_title(list_name[j+i*2], fontsize=12)
        
# %% [markdown]
# 0 residential  
# 1 commercial zones  
# 2 edu  
# 3 green land  
# 4 public service  