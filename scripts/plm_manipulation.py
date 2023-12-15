from transformers import AutoModel, AutoTokenizer
import torch, pandas as pd, numpy as np
import re
import ankh, inspect

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def start_ablang(ablang_model: str):
    tokenizer = AutoTokenizer.from_pretrained(f'qilowoq/{ablang_model}')
    model = AutoModel.from_pretrained(f'qilowoq/{ablang_model}', trust_remote_code=True)
    return model, tokenizer

def start_ankh(device):
    model, tokenizer = ankh.load_large_model()
    model.eval()
    model.to(device=device)
    return model, tokenizer

def chunks(l, n):
    '''
    splits a list into evenly sized chunks
    '''
    return [l[i:i + n] for i in range(0, len(l), n)]

def get_aa_embedding(sequence: str, model: str, tokenizer, max_length: int):
    '''
    Apply fun to a list of sequences using Ablang and generate per-residue embeddings
    '''
    sequence = ' '.join(sequence)
    encoded_input = tokenizer(sequence, padding='max_length', return_tensors='pt')
    model_output = model(**encoded_input)
    lhs = model_output.last_hidden_state
    return lhs.detach().numpy()

def process_seqs(seqs: list, model, tokenizer, device=device) -> list:
    seqs = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in seqs]
    ids = tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids,attention_mask=attention_mask)
    ember = embedding_repr.last_hidden_state
    ember = [i.mean(dim=0) for i in ember]
    return ember

def batch_embed(df: pd, prot_col: str, seq_id: str, batch_size: int, model, tokenizer) -> pd:
    l = []
    batch_size = 100
    for i in range(0, df.shape[0], batch_size):
        df1 = df.iloc[i:i + batch_size]
        batch = df1[prot_col].to_list()
        batch = process_seqs(seqs=batch, model=model, tokenizer=tokenizer)
        protein_embeddings_np = np.array([emb.cpu().numpy() for emb in batch])
        df_tensor = pd.DataFrame(protein_embeddings_np)
        df_tensor['seq_id'] = df1[seq_id].to_list()
        l.append(df_tensor)
    return pd.concat(l)
    
def get_sequence_embeddings(encoded_input, model_output):
    '''
    Taken from Ablang paper - may not work on other pLMs
    '''
    mask = encoded_input['attention_mask'].float()
    d = {k: v for k, v in torch.nonzero(mask).cpu().numpy()} # dict of sep tokens
    # make sep token invisible
    for i in d:
        mask[i, d[i]] = 0
    mask[:, 0] = 0.0 # make cls token invisible
    mask = mask.unsqueeze(-1).expand(model_output.last_hidden_state.size())
    sum_embeddings = torch.sum(model_output.last_hidden_state * mask, 1)
    sum_mask = torch.clamp(mask.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def redux_fit(model, components: int, randstate: int, data: pd, **kwargs) -> pd:
    np.random.seed(randstate)
    model_args = inspect.signature(model).parameters
    if 'method' in model_args:
        method = kwargs.pop('method', None)
        if method:
            redux = model(n_components=components, method=method).fit_transform(data)
        else:
            redux = model(n_components=components).fit_transform(data)
    else:
        redux = model(n_components=components).fit_transform(data)
    X=redux[:, 0]
    y=redux[:, 1]
    new_df = pd.DataFrame([X, y]).transpose()
    new_df.rename(columns={0:'X', 1:'y'}, inplace=True)
    return new_df
    
def run_ankh(df: pd, prot_col: str, seq_id: str) -> pd:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using device: {}".format(device))
    
    model, tokenizer = start_ankh(device=device)
    tensor_df = batch_embed(df=df, prot_col=prot_col, seq_id=seq_id, batch_size=100, model=model, tokenizer=tokenizer)
    return tensor_df
    
def run_ablang(df: pd, prot_col: str, seq_id: str, ablang_model: str) -> pd:
    device = 'cpu'
    model, tokenizer = start_ablang(ablang_model=ablang_model)
    tensor_df = batch_embed(df=df, prot_col=prot_col, seq_id=seq_id, batch_size=100, model=model, tokenizer=tokenizer)
    return tensor_df