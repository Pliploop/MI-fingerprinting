from typing import Any
import torch 
# cosine similarity
from torch.nn.functional import cosine_similarity
from fingerprinting.models import *
from fingerprinting.dataloading.loading_utils import load_full_and_split
import os
import pickle
from tqdm import tqdm
from fingerprinting.evaluation.metrics import *
# softmax
from torch.nn.functional import softmax

DEFAULT_ENCODER = SampleCNN
DEFAULT_CHECKPOINT = "checkpoints/samplecnn/default.ckpt"

# or 

# DEFAULT_ENCODER = VGGish
# DEFAULT_CHECKPOINT = "checkpoints/vggish/default.ckpt"

class FingerPrint(torch.Tensor):
    """ a wrapper around tensors with comparison and quantization functions"""
    
    def __new__(cls, fingerprints, *args, **kwargs):
        if isinstance(fingerprints, torch.Tensor):
            data = fingerprints
        else:
            data = torch.cat(fingerprints)
        return super().__new__(cls, data, *args, **kwargs)

    def sim(self, query: torch.Tensor) -> torch.Tensor:
        # both are tensors of shape [Time1, Features], [Time2, Features]
        # compute similarity over a sliding window
        
        # compute similarity of shape [Time1, Time2]
        cosim = cosine_similarity(self.unsqueeze(1), query.unsqueeze(0),dim = 2)
        return cosim
    
    def compare(self, query: torch.Tensor) -> torch.Tensor:
        sim  = self.sim(query)
        # get the maximum similarity for each query averaged over the sliding window
        timeline =  sim.mean(-1)
        
        return timeline, sim.max().item() # use this for ranking the queries
        
    def quantize(self, quantize_q: int = 100):
        # simple quantization of the fingerprint, inplace
        return FingerPrint(torch.round(self * quantize_q) / quantize_q)
        
        

class fingerPrintBuilder:
    """
    A class for building audio fingerprints from audio files.
    """

    def __init__(self, model=None, feat_extract_head=0, checkpoint=DEFAULT_CHECKPOINT):
        """
        Initializes a fingerPrintBuilder object.

        Args:
            model (torch.nn.Module, optional): The encoder model to use for feature extraction. If None, a default encoder model is used. Defaults to None.
            quantize_q (int, optional): The number of quantization levels for the fingerprints. Defaults to 100.
            feat_extract_head (int, optional): The index of the feature extraction head to use. Defaults to 0.
            checkpoint (str, optional): The path to the checkpoint file for the model. If None, no checkpoint is loaded. Defaults to DEFAULT_CHECKPOINT.
            overlap (float, optional): The overlap ratio for splitting audio into segments. Defaults to 0.5.
        """
        if model is None:
            self.model = ContrastiveFingerprint(encoder=DEFAULT_ENCODER(), feat_extract_head=feat_extract_head)
        else:
            self.model = ContrastiveFingerprint(encoder=model(), feat_extract_head=feat_extract_head)

        #get available device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if checkpoint is not None:
            self.model.load_state_dict(torch.load(checkpoint,map_location = self.device)['state_dict'], strict=True)
            print(f"Loaded model from {checkpoint}")

        self.model.freeze()
        self.model.eval()

        self.target_sr = self.model.encoder.sr
        print(f"Target sample rate: {self.target_sr}")
        self.target_n_samples = self.model.encoder.n_samples
        print(f"Target number of samples: {self.target_n_samples}")


    def get_audio_fingerprints(self, path_to_audio, quantize_q = None, overlap = 0.5) -> FingerPrint:
        """
        Extracts audio fingerprints from the given audio file.

        Args:
            path_to_audio (str): The path to the audio file.

        Returns:
            FingerPrint: The extracted audio fingerprints.
        """
        audio = load_full_and_split(path_to_audio, self.target_sr, self.target_n_samples, overlap=overlap)

        if audio is None:
            return None

        fingerprints = self.model.extract_features(audio.to(self.device))['encoded']
        

        fingerprint = FingerPrint(fingerprints).cpu()
        if quantize_q:
            fingerprint = fingerprint.quantize(quantize_q)
            

        return fingerprint,audio

    def __call__(self, path_to_database, path_to_fingerprints = None, return_=False, save=True, quantize_q = None, overlap = 0.5) -> Any:
        """
        Loads the audio files in path_to_database and saves the fingerprints in path_to_fingerprints.

        Args:
            path_to_database (str): The path to the directory containing the audio files.
            path_to_fingerprints (str): The path to the directory where the fingerprints will be saved.
            return_ (bool, optional): Whether to return the fingerprints as a dictionary. Defaults to False.
            save (bool, optional): Whether to save the fingerprints. Defaults to True.

        Returns:
            Any: The fingerprints dictionary, if return_ is True.
        """
        db = {}

        # scan the directory and get all the audio files as relative paths
        for root, dirs, files in os.walk(path_to_database):
            for file in tqdm(files):
                if file.endswith('.wav') or file.endswith('.mp3'):
                    fingie,_ = self.get_audio_fingerprints(os.path.join(root, file), quantize_q=quantize_q, overlap=overlap)
                    db[file] = fingie if fingie is not None else print(f"Warning: {file} is too short to build a fingerprint.")

        # save the fingerprints
        if save and path_to_fingerprints is not None:
            if not os.path.exists(path_to_fingerprints):
                os.makedirs(path_to_fingerprints)
            # for key in db.keys():
            #     torch.save(db[key], os.path.join(path_to_fingerprints, key + '.pt'))
        
            with open(os.path.join(path_to_fingerprints, 'fingerprints.pkl'), 'wb') as file:
                pickle.dump(db, file)


        if return_:
            return db
        

class audioIdentification:
    """
    Class for audio identification using fingerprinting technique.
    
    Args:
        model (str): Path to the pre-trained model (default: None).
        checkpoint (str): Path to the model checkpoint (default: DEFAULT_CHECKPOINT).
        feat_extract_head (int): Number of layers to extract features from (default: 0).
        overlap (float): Overlap ratio for fingerprinting (default: 0.5).
        quantize_q (int): Quantization factor for fingerprinting (default: 100).
    """
    
    def __init__(self, model=None, checkpoint=DEFAULT_CHECKPOINT, feat_extract_head=0):
        self.builder = fingerPrintBuilder(model=model, checkpoint=checkpoint, feat_extract_head=feat_extract_head)
        
    def __call__(self, path_to_queryset, path_to_fingerprints, path_to_output = None, k=3, return_=False, save=True, quantize_q = None, overlap = 0.5):
        """
        Perform audio identification using fingerprinting technique.
        
        Args:
            path_to_queryset (str): Path to the query set.
            path_to_fingerprints (str): Path to the fingerprints.
            path_to_output (str): Path to save the output file.
            k (int): Number of top matches to retrieve for each query (default: 3).
            return_ (bool): Whether to return the top k matches (default: False).
            save (bool): Whether to save the top k matches to a file (default: True).
        
        Returns:
            dict or None: Dictionary of top k matches for each query, or None if return_ is False.
        """
        query_set = self.builder(path_to_queryset, return_=True, save=False, quantize_q=quantize_q, overlap=overlap)
        
        if "fingerprints.pkl" in os.listdir(path_to_fingerprints):
            fingerprints = pickle.load(open(os.path.join(path_to_fingerprints, 'fingerprints.pkl'), 'rb'))
        else:
            fingerprints = {}
            for file in os.listdir(path_to_fingerprints):
                if file.endswith('.pt'):
                    fingerprints[file] = torch.load(os.path.join(path_to_fingerprints, file))
        
        # get the top k matches for each query
        
        # edge cases :
        # 1. if the query set is empty
        if len(query_set) == 0:
            print("Warning: Query set is empty")
            return {} if return_ else None
        
        # 2. if the fingerprints are empty
        if len(fingerprints) == 0:
            print("Warning: Fingerprints are empty")
            return {} if return_ else None
        
        # 3. if the fingerprints are less than k
        if k > len(fingerprints):
            print(f"Warning: Number of fingerprints is less than top-k {k}. Setting k to {len(fingerprints)}")
            k = len(fingerprints)
        
        similarities = self.build_similarities(query_set, fingerprints)
        ## get the top k matches
        top_k_matches = self.get_top_k_matches(similarities, k)
            
        # save the top k matches to a .txt file of format query_file_name, match1_file_name, match2_file_name, match3_file_name ...

        # create a file to write the results
        if save and path_to_output:
            with open(path_to_output, 'w') as file:
                # iterate over each query
                for query, matches in top_k_matches.items():
                    # write the query file name
                    file.write(query + '\t')
                    # write the match file names
                    for match in matches:
                        file.write(match[0] + '\t')
                    file.write('\n')
        
        if return_:
            return similarities, fingerprints, top_k_matches
        
    def build_similarities(self,query_set, fingerprints):
        """
        Build the similarities between the query set and the fingerprints.
        
        Args:
            query_set (dict): Dictionary of query fingerprints.
            fingerprints (dict): Dictionary of fingerprints.
        
        Returns:
            dict: Dictionary of similarities between the query set and the fingerprints.
        """
        similarities = {}
        for query in query_set.keys():
            similarities[query] = {}
            for key in fingerprints.keys():
                # here : key.compare(query)
                similarities[query][key] = fingerprints[key].compare(query_set[query])
        return similarities    
    
    def get_top_k_matches(self,similarities, k=3):
        """
        Get the top k matches for each query.
        
        Args:
            similarities (dict): Dictionary of similarities between the query set and the fingerprints.
            k (int): Number of top matches to retrieve for each query (default: 3).
        
        Returns:
            dict: Dictionary of top k matches for each query.
        """
        top_k = {}
        for query in similarities.keys():
            top_k[query] = sorted(similarities[query].items(), key=lambda x: x[1][1], reverse=True)[:k]
        return top_k
    
    def evaluate(self, path_to_queryset, path_to_database, top_k, overlap = 0.5, quantize_q = 100):
        
        sims, fingerprints, topk = self(path_to_queryset, path_to_database, top_k, save=False, return_=True, quantize_q=quantize_q, overlap=overlap)
        ground_truth = build_ground_truth(sims, fingerprints)
        metrics = get_metrics(top_k, sims, ground_truth)
        return metrics
        