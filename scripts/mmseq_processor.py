import pandas as pd
import os
import random
import sys

def pw_generator(chars: int) -> str:
  letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
  numbers = [1,2,3,4,5,6,7,8,9] 
  new_l = [letters, numbers]
  pw = ''
  for i in range(chars):
    choix = random.choice(new_l)
    pw += str(random.choice(choix))
  return pw

print('job started')

arg1 = sys.argv[1]

clus = pd.read_csv('/tmp/clusterRes_cluster.tsv', sep='\t', header=None)

# Create a dictionary to store the cluster numbers
cluster_dict = {}

# Iterate over the rows of the dataframe
for idx, row in clus.iterrows():
    id1 = row[0]
    id2 = row[1]

    # Check if id1 or id2 already have a cluster assigned
    if id1 in cluster_dict and id2 not in cluster_dict:
        cluster_dict[id2] = cluster_dict[id1]
    elif id1 not in cluster_dict and id2 in cluster_dict:
        cluster_dict[id1] = cluster_dict[id2]
    elif id1 not in cluster_dict and id2 not in cluster_dict:
        # Assign a new cluster number
        new_cluster = len(cluster_dict)
        cluster_dict[id1] = new_cluster
        cluster_dict[id2] = new_cluster

# Assign cluster numbers to the original dataframe
clus[f'cluster'] = clus[0].map(cluster_dict)

# Create a single column of unique IDs with respective cluster numbers
l = clus['cluster'].to_list() + clus['cluster'].to_list()
new = pd.concat([clus[0], clus[1]])
new = pd.DataFrame(new)
new[f'cluster_{arg1}'] = l

new.drop_duplicates(subset=0, inplace=True)
new.rename(columns={0:'seq'}, inplace=True)

password = pw_generator(8)

new.to_csv(f'/content/drive/MyDrive/msc_project/mmseqs2_output/230723_mmseq_{arg1}_{password}.csv')

print('job done')
