def __main__():
    import polars as pl, pandas as pd
    import gzip, os, json, time
    import pprint as pp
    
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
            df = pl.read_csv(fs, skip_rows=1)
            df = df[['sequence_id_heavy', 'ANARCI_status_heavy', 'sequence_heavy', 'sequence_alignment_aa_heavy', 'v_call_heavy', 'd_call_heavy', 'j_call_heavy', 'sequence_id_light', 'ANARCI_status_light', 'sequence_light', 'v_call_light', 'j_call_light', 'sequence_alignment_aa_light']]
            #add a column using the metadata key
            Run = pl.Series([metadata['Run']]*df.shape[0]).alias('Run')
            # Add the new column to the DataFrame
            df = df.with_columns(Run)
            #save the dataframe to a list
            l.append(df)
            
    l = [i.with_columns(pl.col('Run').cast(pl.Utf8)) for i in l]
    
    df = pl.concat(l)
    
    df = df.with_columns(
        pl.concat_str([
            pl.col('sequence_id_heavy'),
            pl.lit('_'),
            pl.col('sequence_id_light'),
            pl.lit('_'),
            pl.col('Run')
        ]).alias('seq_id')
    )
    
    # #having to get rid of duplicated seq IDs by adding '_1' to every duplicated sequence
    df = df.to_pandas()
    df['Run'] = df['Run'].astype(str)
    duplicated_mask = df['seq_id'].duplicated()
    
    df['seq_id'] = df.apply(lambda row: row['seq_id'] + '_1' if duplicated_mask[row.name] else row['seq_id'], axis=1)
    
    df = df[~df['ANARCI_status_heavy'].str.contains('Shorter')]
    df = df[~df['ANARCI_status_light'].str.contains('Shorter')]
    
    df.to_csv(f'/content/all_paired/{today}_human_paired_seqs.csv')
    
    for filey in files:
        if 'csv.gz' in filey:
            os.remove(filey)
    
if __name__ == "__main__":
  __main__()