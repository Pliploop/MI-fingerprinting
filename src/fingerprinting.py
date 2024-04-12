from typing import Any
import torch 
# cosine similarity
from torch.nn.functional import cosine_similarity
from src.models import *
from src.dataloading.loading_utils import load_full_and_split
import os
import pickle

DEFAULT_ENCODER = SampleCNN
DEFAULT_CHECKPOINT = "checkpoints/default.ckpt"

class FingerPrint(torch.Tensor):
    """ a wrapper around tensors with comparison and quantization functions"""
    
    def __init__(self, fingerpints):
        if isinstance(fingerpints, torch.Tensor):
            self.fingerprints = fingerpints
        else:
            self.fingerprints = torch.cat(fingerpints)
        super().__init__(self.fingerprints)
        self.similarity = cosine_similarity
        
    def sim(self, query: torch.Tensor) -> torch.Tensor:
        # both are tensors of shape [Time1, Features], [Time2, Features]
        # compute similarity over a sliding window
        
        # compute similarity of shape [Time1, Time2]
        return self.similarity(self.fingerprints.unsqueeze(1), query.unsqueeze(0),dim = 2)
    
    def compare(self, query: torch.Tensor) -> torch.Tensor:
        sim  = self.sim(query)
        # get the maximum similarity for each query averaged over the sliding window
        return sim.mean(-1).max(-1) # use this for ranking the queries
        
    def quantize(self, quantize_q: int = 100):
        # simple quantization of the fingerprint, inplace
        self.fingerprints = torch.round(self.fingerprints * quantize_q) / quantize_q
        
        

class FingerPrintBuilder:
    
    """A class which builds a fingerprint for audio. uses a model to extract FingerPrints and stores them in a DataSet."""
    
    def __init__(self, model = None, quantize_q = 100, feat_extract_head = 0, checkpoint = DEFAULT_CHECKPOINT, overlap = 0.5):
        if model is None:
            model = ContrastiveFingerprint(encoder = DEFAULT_ENCODER(), feat_extract_head=feat_extract_head)
        else:
            model = ContrastiveFingerprint(encoder = model, feat_extract_head=feat_extract_head)
        
        if checkpoint is not None:
            model.load_state_dict(torch.load(checkpoint)['state_dict'])
            
        model.freeze()
        
        self.target_sr = model.encoder.sr
        self.target_n_samples = model.encoder.n_samples
        self.overlap = overlap
        
        self.quantize_q = quantize_q
    
    def get_audio_fingerprints(self, path_to_audio) -> FingerPrint:
        """Extracts fingerprints from audio files"""
        
        audio = load_full_and_split(path_to_audio, self.target_sr, self.target_n_samples, self.overlap)
        
        fingerprints = self.model.extract_features(audio)['encoded']
        
        fingerprint = FingerPrint(fingerprints)
        if self.quantize_q:
            fingerprint.quantize(self.quantize_q)
        
        return fingerprint
     
    def __call__(self, path_to_database, path_to_fingerprints, return_ = False, save= True) -> Any:
        
        ## loads the audio files in path_to_database and saves the fingerprints in path_to_fingerprints
        # save fingerprints as both a numpy array and save the fingerprints dictionary as a pickle file
        
        db = {}
        
        # scan the directory and get all the audio files as relative paths
        for root, dirs, files in os.walk(path_to_database):
            for file in files:
                if file.endswith('.wav') or file.endswith('.mp3'):
                    db[file] = self.get_audio_fingerprints(os.path.join(root, file))
        
        # save the fingerprints
        if save:
            for key in db.keys():
                torch.save(db[key], os.path.join(path_to_fingerprints, key + '.pt'))
            
            pickle.dump(db, open(os.path.join(path_to_fingerprints, 'fingerprints.pkl'), 'wb'))
        
        if return_:
            return db
        

class AudioIdentification:
    
    """a callable class that when called with path_to_queryset, path_to_fingerprints, returns the top k matches for each query in the query set
    The query set is a folder containing audio files"""
    
    def __init__(self, model = None, checkpoint = DEFAULT_CHECKPOINT, feat_extract_head = 0, overlap = 0.5, quantize_q = 100):
        
        self.builder = FingerPrintBuilder(model = model, checkpoint = checkpoint, feat_extract_head = feat_extract_head, overlap = overlap, quantize_q = quantize_q)
        
    def __call__(self, path_to_queryset, path_to_fingerprints, path_to_output, k = 3, return_ = False, save = True):
        
        query_set = self.builder(path_to_queryset, path_to_fingerprints, return_ = True)
        
        if "fingerprints.pkl" in os.listdir(path_to_fingerprints):
            fingerprints = pickle.load(open(os.path.join(path_to_fingerprints, 'fingerprints.pkl'), 'rb'))
        else:
            fingerprints = {}
            for file in os.listdir(path_to_fingerprints):
                if file.endswith('.pt'):
                    fingerprints[file] = torch.load(os.path.join(path_to_fingerprints, file))
        
        # get the top k matches for each query
        top_k = {}
        for query in query_set.keys():
            top_k[query] = {}
            for key in fingerprints.keys():
                top_k[query][key] = fingerprints[key].compare(query_set[query])
        
        ## get the top k matches
        top_k_matches = {}
        for query in top_k.keys():
            top_k_matches[query] = sorted(top_k[query].items(), key = lambda x: x[1], reverse = True)[:k]
            
        # save the top k matches to a .txt file of format query_file_name, match1_file_name, match2_file_name, match3_file_name ...

        # create a file to write the results
        if save:
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
            return top_k_matches

# wrap both above class intanciations and calls within two methods fingerPrintBuilder and audioIdentification

def fingerprintBuilder(path_to_database, path_to_fingerprints, model = None, checkpoint = DEFAULT_CHECKPOINT, feat_extract_head = 0, overlap = 0.5, quantize_q = 100, return_ = False, save = True):
    builder = FingerPrintBuilder(model = model, checkpoint = checkpoint, feat_extract_head = feat_extract_head, overlap = overlap, quantize_q = quantize_q)
    return builder(path_to_database, path_to_fingerprints, return_ = return_, save = save)

def audioIdentification(path_to_queryset, path_to_fingerprints, path_to_output, model = None, checkpoint = DEFAULT_CHECKPOINT, feat_extract_head = 0, overlap = 0.5, quantize_q = 100, k = 3, return_ = False, save = True):
    identification = AudioIdentification(model = model, checkpoint = checkpoint, feat_extract_head = feat_extract_head, overlap = overlap, quantize_q = quantize_q)
    return identification(path_to_queryset, path_to_fingerprints, path_to_output, k = k, return_ = return_, save = save)