import torch
import torch.nn as nn
from supar.config import Config
from supar.models.dep.biaffine.model import BiaffineDependencyModel


class GNNDependencyModel(BiaffineDependencyModel):
    def __init__(self, 
                 n_words, 
                 n_rels, 
                 n_tags=None,
                 n_chars=None, 
                 encoder='lstm', 
                 feat=['char'], 
                 n_embed=100, 
                 n_pretrained=100, 
                 n_feat_embed=100, 
                 n_char_embed=50, 
                 n_char_hidden=100, 
                 char_pad_index=0, 
                 elmo='original_5b', 
                 elmo_bos_eos=(True, False), 
                 bert=None, 
                 n_bert_layers=4, 
                 mix_dropout=0, 
                 bert_pooling='mean', 
                 bert_pad_index=0, 
                 finetune=False, 
                 n_plm_embed=0, 
                 embed_dropout=0.33, 
                 n_encoder_hidden=800, 
                 n_encoder_layers=3, 
                 encoder_dropout=0.33, 
                 n_arc_mlp=500, 
                 n_rel_mlp=100, 
                 mlp_dropout=0.33, 
                 scale=0, 
                 pad_index=0, 
                 unk_index=1, 
                 **kwargs):
        super().__init__(**Config().update(locals()))

        #init GNN layer
    
    def forward(self, words, feats=None):
        pass

    def embed(self, words, feats=None):
        ext_words = words
        # set the indices larger than num_embeddings to unk_index
        if hasattr(self, 'pretrained'):
            ext_mask = words.ge(self.word_embed.num_embeddings)
            ext_words = words.masked_fill(ext_mask, self.args.unk_index)

        # get outputs from embedding layers
        word_embed = self.word_embed(ext_words)
        if hasattr(self, 'pretrained'):
            pretrained = self.pretrained(words)
            if self.args.n_embed == self.args.n_pretrained:
                word_embed += pretrained
            else:
                word_embed = torch.cat((word_embed, self.embed_proj(pretrained)), -1)
                
        feat_embed = []
        if 'tag' in self.args.feat:
            feat_embed.append(self.tag_embed(feats.pop()))
        if 'char' in self.args.feat:
            feat_embed.append(self.char_embed(feats.pop(0)))
        if 'elmo' in self.args.feat:
            feat_embed.append(self.elmo_embed(feats.pop(0)))
        if 'bert' in self.args.feat:
            feat_embed.append(self.bert_embed(feats.pop(0)))
        if 'lemma' in self.args.feat:
            feat_embed.append(self.lemma_embed(feats.pop(0)))