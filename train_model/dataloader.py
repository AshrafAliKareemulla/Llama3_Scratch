from torch.utils.data import DataLoader
import pandas as pd


def dataloader_fun():
    path = "../ita-eng/ita.txt"

    data = pd.read_csv(path, sep='\t')
    
    data.columns = ["english", "italian", "attrib"]
    data.drop(["attrib"], inplace=True, axis=1)

    
