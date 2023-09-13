import os

def opig_all_paired():
    os.system("wget -P /content/all_paired/opig_data -i /content/msc-project-source-code-files-22-23-mbalmf01/data_files/wget_commands.txt")

def clean_human_abs():
    os.system("python /content/msc-project-source-code-files-22-23-mbalmf01/scripts/data_preparation.py")

def install_miniconda():
    os.system("wget -P /content/all_paired/opig_data -i /content/msc-project-source-code-files-22-23-mbalmf01/scripts/wget_commands.txt")
    os.system("wget https://repo.anaconda.com/miniconda/Miniconda3-py39_23.3.1-0-Linux-x86_64.sh")
    os.system("chmod +x Miniconda3-py39_23.3.1-0-Linux-x86_64.sh")
    os.system("echo 'y' | bash ./Miniconda3-py39_23.3.1-0-Linux-x86_64.sh -b -f -p /usr/local")

def install_mmseqs2():
    os.system("conda install -c conda-forge -c bioconda mmseqs2")

def __main__():
    import pandas as pd
    import gzip, os, json, time
    import pprint as pp
    
    print('Running...')
    # Get the current date
    date = time.strftime('%D')
    today = date[-2:] + date[:2] + date[3:5]
    
    os.chdir('/content/all_paired/opig_data')
    files = os.listdir()
    
    l = []
    for filey in files:
      if 'csv.gz' in filey:
        #filey = files[1]
        with gzip.open(filey, 'rb') as f:
          #extract metadata and save as a dictionary
          metadata = f.readline()
          metadata = metadata.decode()
          metadata = metadata.strip().strip('"').replace('""', '"')
          metadata = json.loads(metadata)
    
        if metadata['Species'] == 'human':
          with gzip.open(filey, 'rb') as fs:
            # #read in antibody data and keep id, dna, and protein sequences
            df = pd.read_csv(fs, skiprows=1, low_memory=False)
            df = df[['sequence_id_heavy', 'ANARCI_status_heavy', 'sequence_heavy', 'sequence_alignment_aa_heavy', 'v_call_heavy', 'd_call_heavy', 'j_call_heavy', 'sequence_id_light', 'ANARCI_status_light', 'sequence_light', 'v_call_light', 'j_call_light', 'sequence_alignment_aa_light']]
            #add a column using the metadata key
            Run = [metadata['Run']]*df.shape[0]
            df['Run'] = Run
            #save the dataframe to a list
            l.append(df)
            
    df = pd.concat(l)
    df['Run'] = df['Run'].astype(str)

    df['seq_id'] = (df['sequence_id_heavy'] 
          + ['_']*df.shape[0] 
          + df['sequence_id_light'] 
          + ['_']*df.shape[0] 
          + df['Run'])

    #having to get rid of duplicated seq IDs by adding '_1' to every duplicated sequence
    duplicated_mask = df['seq_id'].duplicated()
    suffix = duplicated_mask.astype(int).astype(str).replace('0', '')

    #concatenate the original sequence IDs with the suffix series
    df['seq_id'] = df['seq_id'] + suffix
    total = df.shape[0]
    print(f'Dataframe contains {total} antibodies before data cleaning')
    df = df[~df['ANARCI_status_heavy'].str.contains('Shorter')]
    df = df[~df['ANARCI_status_light'].str.contains('Shorter')]

    print(f'{total - df.shape[0]} antibodies were removed due to truncations')
    print(f'writing {df.shape[0]} antibodies to file...')
    
    df.to_csv(f'/content/all_paired/{today}_human_paired_seqs.csv')
    
if __name__ == "__main__":
  __main__()