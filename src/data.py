import json
import codecs 
import functools
from tqdm import tqdm  
import numpy as np 
from pathlib import Path 
from typing import Tuple, Union, List, Dict
from collections import Counter
import operator
import unicodedata
import logging
import pickle 
import torch 
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import sentencepiece as spm 
from transformers import RobertaTokenizer
import re  
import os 

"""
Global constants
"""
UNK_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"

UNK_ID = 0
PAD_ID = 1
BOS_ID = 2
EOS_ID = 3

logger = logging.getLogger(__name__)

class Vocabulary(object):
    """
    Vocabulary class mapping between tokens and indices.
    """
    def __init__(self, tokens: List[str]) -> None:
        "Create  vocabulary from list of tokens. :param tokens: list of tokens"

        self.specials = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
        self._stoi: Dict[str, int] = {} # string to index
        self._itos: List[str] = []      # index to string

        # construct vocabulary
        self.add_tokens(tokens=self.specials + tokens)
        assert len(self._stoi) == len(self._itos)

        # assign special after stoi and itos are built
        self.pad_index = self.lookup(PAD_TOKEN)
        self.unk_index = self.lookup(UNK_TOKEN)
        self.bos_index = self.lookup(BOS_TOKEN)
        self.eos_index = self.lookup(EOS_TOKEN)
        assert self.pad_index == PAD_ID
        assert self.unk_index == UNK_ID
        assert self.bos_index == BOS_ID
        assert self.eos_index == EOS_ID
        assert self._itos[UNK_ID] == UNK_TOKEN
    
    def lookup(self, token: str) -> int:
        "look up the encoding dictionary"
        return self._stoi.get(token, UNK_ID) 
    
    def add_tokens(self, tokens:List[str]) -> None:
        for token in tokens:
            token = self.normalize(token)
            new_index = len(self._itos)
            # add to vocabulary if not already there
            if token not in self._itos:
                self._itos.append(token)
                self._stoi[token] = new_index
    
    def is_unk(self,token:str) -> bool:
        """
        Check whether a token is covered by the vocabulary.
        """
        return self.lookup(token) == UNK_ID
    
    def to_file(self, file_path: Path) -> None:
        def write_list_to_file(file_path:Path, array:List[str]) -> None:
            """
            write list of str to file.
            """
            with file_path.open("w", encoding="utf-8") as fg:
                for item in array:
                    fg.write(f"{item}\n")
        
        write_list_to_file(file_path, self._itos)
    
    def __len__(self) -> int:
        return len(self._itos)
    
    @staticmethod
    def normalize(token) -> str:
        return unicodedata.normalize('NFD', token)
    
    def array_to_sentence(self, array: np.ndarray, cut_at_eos: bool=True, skip_pad: bool=True) -> List[str]:
        """
        Convert an array of IDs to a sentences (list of tokens).
        array: 1D array containing indices
        Note: when cut_at_eos=True, sentence final token is </s>.
        """
        sentence = []
        for i in array:
            token = self._itos[i]
            if skip_pad and token == PAD_TOKEN:
                continue
            sentence.append(token)
            if cut_at_eos and token == EOS_TOKEN:
                break
        
        return [token for token in sentence if token not in self.specials]

    def arrays_to_sentences(self, arrays: np.ndarray, cut_at_eos: bool=True, skip_pad: bool=True) -> List[List[str]]:
        """
        Convert multiple arrays containing sequences of token IDs to their sentences.
        arrays: 2D array containing indices.
        return: list of list of tokens.
        """
        return [self.array_to_sentence(array=array, cut_at_eos=cut_at_eos, skip_pad=skip_pad) for array in arrays]

    def tokens_to_ids(self, tokens:List[str], bos:bool=False, eos:bool=False):
        """
        Return sentences_ids List[id].
        """
        tokens_ids = [self.lookup(token) for token in tokens]
        if bos is True:
            tokens_ids = [self.bos_index] + tokens_ids
        if eos is True:
            tokens_ids = tokens_ids + [self.eos_index]

        return tokens_ids

    def log_vocab(self, number:int) -> str:
        "First how many number of tokens in Vocabulary."
        return " ".join(f"({id}) {token}" for id, token in enumerate(self._itos[:number]))

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(len={self.__len__()}, "
                f"specials={self.specials})")

def read_data_from_file(path:Path, tokenizer_src=None, tokenizer_trg=None, tokenizer_original=None):
    with codecs.open(path, encoding='utf-8') as f:
        raw_data = json.load(f)
    
    data = []
    dataset_truth = []
    for item in tqdm(raw_data, desc="Extract data from file..."):
        instruct_order = str()
        for each_instruct in item['function_body']:
            # each_instruct = re.sub(r'[\da-fA-F]{4,}', '', each_instruct)
            instruct_order = instruct_order + ' ' + each_instruct
        if tokenizer_src is not None:
            # assembly_tokens = tokenizer.EncodeAsPieces(instruct_order)
            instruct_order = re.sub(r'[\da-fA-F]{4,}', '', instruct_order)
            assembly_tokens = tokenizer_src.tokenize(instruct_order)
        else:
            instruct_order = re.sub(r'[\da-fA-F]{4,}', '', instruct_order)
            assembly_tokens = instruct_order.strip().split()

        comment_truth = item["comment"]
        dataset_truth.append(comment_truth)

        if tokenizer_trg is not None:
            # comment_tokens = tokenizer.EncodeAsPieces(item['comment'])
            comment_tokens = tokenizer_src.tokenize(item["comment"])
        else:
            comment_tokens = item['comment'].split()

        pseudo_code = item["pseudo_code_refined"]
        if tokenizer_src is not None:
            pseudo_code_tokens = tokenizer_src.tokenize(pseudo_code)
            pseudo_code_tokens_id = tokenizer_original.encode(pseudo_code, max_length=400, padding='max_length', truncation=True)
        else:
            pseudo_code_tokens = pseudo_code.split()

        cfg_nodes = item['cfg']['nodes']

        cfg_edges = np.array(item['cfg']['edge_index'])
        reversed_edges = np.array([cfg_edges[1,:], cfg_edges[0,:]])
        cfg_edges = np.concatenate([cfg_edges, reversed_edges], axis=-1)

        data_item = {
            "assembly_tokens": assembly_tokens,
            "comment_tokens": comment_tokens,
            "pseudo_code_tokens": pseudo_code_tokens,
            "pseudo_code_tokens_id": pseudo_code_tokens_id,
            "cfg_nodes": cfg_nodes,
            "cfg_edges": cfg_edges,
        }
        data.append(data_item)

    return data, dataset_truth

def log_vocab_info(assembly_token_vocab, comment_token_vocab, cfg_node_vocab, pseudo_token_vocab):
    """logging vocabulary information"""
    logger.info("*"*10+"logging vocabulary information"+"*"*10)
    logger.info("assembly token vocab length = {}".format(len(assembly_token_vocab)))
    logger.info("comment token vocab length = {}".format(len(comment_token_vocab)))
    logger.info("pseudo token vocab length = {}".format(len(pseudo_token_vocab)))
    logger.info("cfg node vocab length = {}".format(len(cfg_node_vocab)))

def log_data_info(assembly_tokens, comment_tokens, cfg_nodes, cfg_edges, pseudo_tokens):
    """logging data statistics"""
    assert len(assembly_tokens) == len(comment_tokens) == len(cfg_nodes) == len(cfg_edges) == len(pseudo_tokens), "Data need double checked!"
    len_assembly_tokens = len_comment_tokens = len_cfg_nodes = len_cfg_edges = len_pseudo_tokens = 0
    for assembly_token, comment_token, cfg_node, cfg_edge, pseudo_token in zip(assembly_tokens, comment_tokens, cfg_nodes, cfg_edges, pseudo_tokens):
        len_assembly_tokens += len(assembly_token)
        len_comment_tokens += len(comment_token)
        len_pseudo_tokens += len(pseudo_token)
        len_cfg_nodes += len(cfg_node)
        len_cfg_edges += cfg_edge.shape[1]
    
    logger.info("*"*10+"logging data statistics"+"*"*10)
    logger.info("average assembly tokens = {}".format(len_assembly_tokens / len(assembly_tokens)))
    logger.info("average comment tokens = {}".format(len_comment_tokens / len(comment_tokens)))
    logger.info("average pseudo tokens = {}".format(len_pseudo_tokens / len(pseudo_tokens)))
    logger.info("average cfg nodes = {}".format(len_cfg_nodes / len(cfg_nodes)))
    logger.info("average cfg edges = {}".format(len_cfg_edges / len(cfg_edges)))

def build_vocabulary(data_cfg:dict, datasets):
    def flatten(array):
        # flatten a nested 2D list.
        return functools.reduce(operator.iconcat, array, [])

    def sort_and_cut(counter, max_size:int, min_freq:int) -> List[str]:
        """
        Cut counter to most frequent, sorted numerically and alphabetically.
        return: list of valid tokens
        """
        if min_freq > -1:
            counter = Counter({t: c for t, c in counter.items() if c >= min_freq})
        
        # sort by frequency, then alphabetically
        tokens_and_frequencies = sorted(counter.items(),key=lambda tup: tup[0])
        tokens_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        # cut off
        vocab_tokens = [i[0] for i in tokens_and_frequencies[:max_size]]
        assert len(vocab_tokens) <= max_size, "vocab tokens must <= max_size."
        return vocab_tokens
    
    assembly_token_min_freq = data_cfg['assembly_token'].get("vocab_min_freq", 1)
    assembly_max_size = data_cfg['assembly_token'].get("vocab_max_size", -1)
    assert assembly_max_size > 0 and assembly_token_min_freq > 0
    comment_min_freq = data_cfg['comment'].get("vocab_min_freq", 1)
    comment_max_size = data_cfg['comment'].get("vocab_max_size", -1)
    assert comment_max_size > 0 and comment_min_freq > 0
    cfg_node_min_freq = data_cfg["cfg_node"].get("vocab_min_freq", 1)
    cfg_node_max_size = data_cfg["cfg_node"].get("vocab_max_size", -1)
    assert cfg_node_max_size > 0 and cfg_node_min_freq > 0
    pseudo_token_min_freq = data_cfg["pseudo_token"].get("vocab_min_freq", 1)
    pseudo_token_max_size = data_cfg['pseudo_token'].get("vocab_max_size", -1)
    assert pseudo_token_max_size > 0 and pseudo_token_min_freq > 0

    assembly_tokens = []  # list of list [[], [], []]  already double checked!
    comment_tokens = []  # list of list [[], [], []]
    pseudo_tokens = [] # list of list [[], [], []]
    cfg_nodes = []   # list of list [[], [], []]
    cfg_edges = [] # list of ndarray [ndarray, ndarray, ndarray]
    for idx, dataset in enumerate(datasets):
        for item in tqdm(dataset, desc='build vocabulary for #{} dataset...'.format(idx)):
            assembly_tokens.append(item['assembly_tokens'])
            comment_tokens.append(item['comment_tokens'])
            cfg_nodes.append(item['cfg_nodes'])
            cfg_edges.append(item['cfg_edges'])
            pseudo_tokens.append(item['pseudo_code_tokens'])

    log_data_info(assembly_tokens, comment_tokens, cfg_nodes, cfg_edges, pseudo_tokens)

    assembly_token_counter = Counter(flatten(assembly_tokens))
    comment_token_counter = Counter(flatten(comment_tokens))
    cfg_node_counter = Counter(flatten(cfg_nodes))
    pseudo_token_counter = Counter(flatten(pseudo_tokens))

    assembly_token_unique_tokens = sort_and_cut(assembly_token_counter, assembly_max_size, assembly_token_min_freq)
    comment_token_unique_tokens = sort_and_cut(comment_token_counter, comment_max_size, comment_min_freq)
    cfg_node_unique_tokens = sort_and_cut(cfg_node_counter, cfg_node_max_size, cfg_node_min_freq)
    pseudo_token_unique_tokens = sort_and_cut(pseudo_token_counter, pseudo_token_max_size, pseudo_token_min_freq)

    assembly_token_vocab = Vocabulary(assembly_token_unique_tokens)
    comment_token_vocab = Vocabulary(comment_token_unique_tokens)
    cfg_node_vocab = Vocabulary(cfg_node_unique_tokens)
    pseudo_token_vocab = Vocabulary(pseudo_token_unique_tokens)

    # assembly_token_vocab.to_file(Path("assembly_token.vocab"))
    # comment_token_vocab.to_file(Path("comment_token.vocab"))

    log_vocab_info(assembly_token_vocab, comment_token_vocab, cfg_node_vocab, pseudo_token_vocab)
    return assembly_token_vocab, comment_token_vocab, cfg_node_vocab, pseudo_token_vocab

def load_data(data_cfg: dict):
    """
    Load train, valid and test data as specified in configuration.
    """
    cached_dataset_path = "{}/{}.pt".format(data_cfg["data_path"], data_cfg["cached_dataset_path"])
    if os.path.exists(cached_dataset_path):
        logger.info("Use datset already stored!")
        dataset_stored = torch.load(cached_dataset_path)
        train_dataset = dataset_stored["train_datasest"]
        valid_dataset = dataset_stored["valid_dataset"]
        test_dataset = dataset_stored["test_dataset"]
        vocab_info = dataset_stored["vocab_info"]
        return train_dataset, valid_dataset, test_dataset, vocab_info

    train_data_path = Path(data_cfg.get("train_data_path", None))
    valid_data_path = Path(data_cfg.get("valid_data_path", None))
    test_data_path = Path(data_cfg.get("test_data_path", None))
    assert train_data_path.is_file() and valid_data_path.is_file() and test_data_path.is_file(), \
    "train or valid or test path is not a file."

    use_tokenizer = data_cfg.get("use_tokenizer", None)
    if use_tokenizer == "sentencepiece":
        spm_model_path = data_cfg.get("sentencepiece_binary_model")
        tokenizer = spm.SentencePieceProcessor(model_file=spm_model_path)
    elif use_tokenizer == "robertatokenizer":
        tokenizer_original = RobertaTokenizer.from_pretrained(data_cfg.get("robertatokenizer"))
        tokenizer_src = RobertaTokenizer.from_pretrained(data_cfg.get("robertatokenizer"))
        if data_cfg.get("architecture") == "x86_64" or data_cfg.get("architecture") == "x86_32":
            special_tokens_dict = {'additional_special_tokens': ['<NEAR>', '<FAR>', '<ICALL>', '<ECALL>',
                                                        '<POSITIVE>', '<NEGATIVE>', '<ZERO>', '<VOID>', '<SPECIAL>'
                                                        'r15', 'r14', 'r13', 'r12', 'r11', 'r10', 'r9', 'r8',
                                                        'rbp', 'rsp', 'rdi', 'rsi', 'rax', 'rbx', 'rcx', 'rdx', 
                                                        'eax', 'ebx', 'ecx', 'edx', 'edi', 'esi', 'ebp', 'esp',
                                                        'qword', 'lea', 'call', 'test', 'byte', 'ptr', 'jz', 'jnz', 'jle',
                                                        'push','mov', 'add', 'movzx', 'and', 'shr', 'retn', 'cmp']}
        else:
            special_tokens_dict = {'additional_special_tokens':['<NEAR>', '<FAR>', '<ICALL>', '<ECALL>',
                                                        '<POSITIVE>', '<NEGATIVE>', '<ZERO>', '<VOID>', '<SPECIAL>',
                                                        'ADD', 'SUB', 'AND', 'OR', 'XOR', 'B', 'BL', 'BEQ', 'BNE',
                                                        'BMI', 'BPL', 'LDR', 'STR', 'SWI', 'MRS', 'MSR', 'MRC', 'MCR', 'POP'
                                                        'R0', 'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8',
                                                        'R9', 'R10', 'R11', 'R12', 'R13', 'R14', 'R15']}
        tokenizer_src.add_special_tokens(special_tokens_dict)
        # tokenizer_trg = RobertaTokenizer.from_pretrained(data_cfg.get("robertatokenizer"))
        tokenizer_trg = None 
    else:
        logger.info("not use tokenizer")
        tokenizer_src = None  
        tokenizer_trg = None 

    use_refined_pseudo_code = data_cfg.get("use_refined_pseudo_code", False)
    train_data, train_data_truth = read_data_from_file(train_data_path, tokenizer_src, tokenizer_trg, tokenizer_original, use_refined_pseudo_code)
    valid_data, valid_data_truth = read_data_from_file(valid_data_path, tokenizer_src, tokenizer_trg, tokenizer_original, use_refined_pseudo_code)
    test_data, test_data_truth = read_data_from_file(test_data_path, tokenizer_src, tokenizer_trg, tokenizer_original, use_refined_pseudo_code)

    assembly_token_vocab, comment_token_vocab, cfg_node_vocab, pseudo_token_vocab = build_vocabulary(data_cfg, [train_data, valid_data])

    vocab_info = {
        "assembly_token_vocab": {"self":assembly_token_vocab ,"size": len(assembly_token_vocab), "pad_index": assembly_token_vocab.pad_index},
        "comment_token_vocab": {"self":comment_token_vocab, "size":len(comment_token_vocab), "pad_index": comment_token_vocab.pad_index},
        "cfg_node_vocab": {"self":cfg_node_vocab, "size":len(cfg_node_vocab), "pad_index": cfg_node_vocab.pad_index},
        "pseudo_token_vocab":{"self":pseudo_token_vocab, "size":len(pseudo_token_vocab), "pad_index":pseudo_token_vocab.pad_index},
        "tokenizer_src": tokenizer_src, "tokenizer_trg":tokenizer_trg
    }

    train_data_id = token2id(train_data, assembly_token_vocab, comment_token_vocab, cfg_node_vocab, pseudo_token_vocab, data_cfg)
    valid_data_id = token2id(valid_data, assembly_token_vocab, comment_token_vocab, cfg_node_vocab, pseudo_token_vocab, data_cfg)
    test_data_id = token2id(test_data, assembly_token_vocab, comment_token_vocab, cfg_node_vocab, pseudo_token_vocab, data_cfg)

    train_dataset = BinaryDataset(train_data_id, train_data_truth)
    valid_dataset = BinaryDataset(valid_data_id, valid_data_truth)
    test_dataset = BinaryDataset(test_data_id, test_data_truth)

    dataset_stored = {"train_datasest": train_dataset, "valid_dataset":valid_dataset,
                      "test_dataset": test_dataset, "vocab_info": vocab_info}
    torch.save(dataset_stored, cached_dataset_path)
    logger.info("Store dataset.pt to datas!")
    
    return train_dataset, valid_dataset, test_dataset, vocab_info

def token2id(data, assembly_token_vocab:Vocabulary, comment_token_vocab:Vocabulary, cfg_node_vocab:Vocabulary, pseudo_token_vocab:Vocabulary, data_cfg:dict):
    """
    token to id. And truc and pad for code_tokens_id and text_tokens_id.
    """
    assembly_token_max_len = data_cfg["assembly_token"]["token_max_len"]
    comment_token_max_len = data_cfg["comment"]["token_max_len"]
    pseudo_token_max_len = data_cfg["pseudo_token"]["token_max_len"]
    # data [dict, dict, ...]
    data_id = []
    for item in tqdm(data, desc="token2id"):
        data_item_id = {}
        data_item_id["assembly_tokens_id"] = truc_pad(assembly_token_vocab.tokens_to_ids(item["assembly_tokens"], bos=False, eos=False), assembly_token_max_len)
        data_item_id["pseudo_tokens_id"] = truc_pad(pseudo_token_vocab.tokens_to_ids(item["pseudo_code_tokens"], bos=False, eos=False), pseudo_token_max_len)
        data_item_id["comment_tokens_id"] = truc_pad(comment_token_vocab.tokens_to_ids(item["comment_tokens"], bos=True, eos=True), comment_token_max_len)
        data_item_id["cfg_nodes_id"] = cfg_node_vocab.tokens_to_ids(item["cfg_nodes"], bos=False, eos=False)
        data_item_id["cfg_edgs_id"]= item["cfg_edges"] # ndarray
        data_item_id["pseudo_tokens_codet5_id"] = item["pseudo_code_tokens_id"]
        data_id.append(data_item_id)

    return data_id

def truc_pad(data, token_max_len):
    truc_data = data[:token_max_len]
    pad_number = token_max_len - len(truc_data)
    assert pad_number >=0, "pad number must >=0!"
    pad_data = truc_data + [PAD_ID] * pad_number
    return pad_data

class BinaryDataset(Dataset):
    def __init__(self, data_id, data_truth) -> None:
        super().__init__()
        self.data_id = data_id  # data_id [dict, dict, ...]
        self.target_truth = data_truth
    
    def __getitem__(self, index):
        data_item = self.data_id[index]
        assembly_tokens_id = data_item["assembly_tokens_id"]
        comment_tokens_id = data_item["comment_tokens_id"]
        pseudo_tokens_id = data_item["pseudo_tokens_id"]
        pseudo_tokens_codet5_id = data_item["pseudo_tokens_codet5_id"]
        cfg_nodes_id = data_item["cfg_nodes_id"]
        cfg_edges_id = data_item["cfg_edgs_id"]

        return BinaryData(assembly_tokens=torch.tensor(assembly_tokens_id, dtype=torch.long), 
                       comment_tokens=torch.tensor(comment_tokens_id, dtype=torch.long),
                       pseudo_tokens=torch.tensor(pseudo_tokens_id, dtype=torch.long),
                       pseudo_tokens_codet5 = torch.tensor(pseudo_tokens_codet5_id, dtype=torch.long),
                       cfg_nodes=torch.tensor(cfg_nodes_id, dtype=torch.long), 
                       cfg_edges=torch.tensor(cfg_edges_id, dtype=torch.long))
    
    def __len__(self) -> int:
        return len(self.data_id)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(len={self.__len__()}"

class BinaryData(Data):
    def __init__(self, assembly_tokens, comment_tokens, pseudo_tokens, pseudo_tokens_codet5, cfg_nodes, cfg_edges, *args, **kwargs):
        super().__init__()
        
        self.assembly_tokens = assembly_tokens
        self.comment_tokens = comment_tokens
        self.pseudo_tokens = pseudo_tokens
        self.pseudo_tokens_codet5 = pseudo_tokens_codet5
        self.cfg_nodes = cfg_nodes 
        self.cfg_edges = cfg_edges

    def __inc__(self, key, value, *args, **kwargs):
        if key == "cfg_edges":
            return self.cfg_nodes.size(0)
        else: 
            return 0

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "cfg_edges":
            return -1
        elif key == "assembly_tokens" or key == "comment_tokens" or key == "pseudo_tokens" or key == "pseudo_tokens_codet5":
            return None 
        elif key == "cfg_nodes":
            return 0
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)

def make_data_loader(dataset: BinaryDataset, sampler_seed, shuffle, batch_size, num_workers, mode=None) -> DataLoader:
    """
    Return a torch DataLoader.
    """
    assert isinstance(dataset, Dataset), "For pytorch, dataset is based on torch.utils.data.Dataset"

    if mode == "train" and shuffle is True:
        generator = torch.Generator()
        generator.manual_seed(sampler_seed)
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
    else:
        sampler = SequentialSampler(dataset)
    
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, 
                      num_workers=num_workers, pin_memory=True)

if __name__ == "__main__": 
    logger = logging.getLogger("")
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    logger.info("Hello! This is xxxx's Transformer!")

    import yaml
    def load_config(path: Union[Path,str]="configs/xxx.yaml"):
        if isinstance(path, str):
            path = Path(path)
        with path.open("r", encoding="utf-8") as yamlfile:
            cfg = yaml.safe_load(yamlfile)
        return cfg
    
    cfg_file = "configs/O1_test6_cszx.yaml"
    cfg = load_config(Path(cfg_file))
    train_dataset, valid_dataset, test_dataset, vocab_info = load_data(data_cfg=cfg["data"])

    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=False)
    batch_data = next(iter(train_loader))

    # check batch_data
    logger.info(batch_data)
    logger.info("assembly_tokens = {}".format(batch_data.assembly_tokens))
    logger.info("comment_tokens = {}".format(batch_data.comment_tokens))
    logger.info("cfg_nodes = {}".format(batch_data.cfg_nodes))
    logger.info("cfg_edges = {}".format(batch_data.cfg_edges))
    logger.info("batch = {}".format(batch_data.batch))
    logger.info("ptr = {}".format(batch_data.ptr))