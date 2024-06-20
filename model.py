import torch
import torch.nn as nn
import clip
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from huggingface_hub import hf_hub_download
from torchvision.models import VGG16_Weights

CLIP = 'clip'
VGG = 'vgg'
REMOTE_CLIP = 'remote_clip'

class VisGPT(nn.Module):
    '''
    GPT model equipped with vision capabilities through an image encoder and embedding concatenation.
    '''
    def __init__(self, device, encoder=CLIP, dropout=0.1, train_generator=False, train_adapter=False, train_encoder=False):
        super(VisGPT, self).__init__()

        # Load the GPT2 model and tokenizer
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        gpt2_config.resid_pdrop = dropout if train_generator else 0
        gpt2_config.embd_pdrop = dropout if train_generator else 0
        gpt2_config.attn_pdrop = dropout if train_generator else 0
        self.generator = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.gen_embedding_dim = self.generator.transformer.wte.weight.shape[1]
        
        self.encoder = Encoder(encoder=encoder, device=device)

        self.adapter_layer = Adapter_Layer(encoder_dim = self.encoder.out_dim, embedding_dim=self.gen_embedding_dim, dropout_prob=dropout)

        self.device = device
        
        # Put the various pieces in training/evaluation
        self.set_generator_mode(train_generator)
        self.set_adapter_layer_mode(train_adapter)
        self.set_encoder_mode(train_encoder)
        
    def set_generator_mode(self, trainable:bool):
        for param in self.generator.parameters():
            param.requires_grad = trainable
    
    def set_adapter_layer_mode(self, trainable:bool):
        for param in self.adapter_layer.parameters():
            param.requires_grad = trainable
    
    def set_encoder_mode(self, trainable:bool):
        for param in self.encoder.parameters():
            param.requires_grad = trainable
    

    def train_generator(self, captions, images):
        # Encode images or text to get CLIP embeddings
        decoder_embedding = self.encoder(images)
        # Adapt to the dimensionality of gpt2 embeddings
        decoder_embedding = self.adapter_layer(decoder_embedding)
        
        tokens = self.tokenizer(captions, return_tensors='pt', truncation=False, padding="longest")
        input_ids = tokens.input_ids.to(self.device)
        att_mask = tokens.attention_mask.to(self.device)
        
        gen_embeddings = self.generator.transformer.wte(input_ids)
  
        # Concatenate CLIP embeddings with generator embeddings
        emb_cat = torch.cat([decoder_embedding, gen_embeddings], dim=1)

        # Create an attention mask for the concatenated embeddings
        clip_attention_mask = torch.ones(input_ids.shape[0], decoder_embedding.shape[1]).to(self.device)
        combined_att_mask = torch.cat([clip_attention_mask, att_mask], dim=1)

        with torch.no_grad():
        # set the label corresponding to the CLIP embedding to -100
            labels_img_tokens = torch.fill(torch.ones(input_ids.shape[0], decoder_embedding.shape[1]).long(), -100).to(self.device)
            input_ids[~att_mask.bool()] = -100
            labels = torch.cat([labels_img_tokens, input_ids], dim=1)

        return self.generator(
            inputs_embeds=emb_cat, 
            attention_mask=combined_att_mask,
            labels=labels
        ).loss

    def get_caption(self, images, max_new_tokens=128, k=5):
        """
        Generate captions for the images using the generator
        
        Args:
            images (torch.Tensor): a batch of images
        
        Returns:
            List[str]: captions for the images
        """
        decoder_embedding = self.encoder(images)
        decoder_embedding = self.adapter_layer(decoder_embedding)

        # adding positional embeddings, I need them only here
        pos_emb = self.generator.transformer.wpe(torch.arange(decoder_embedding.shape[1]).to(self.device))
        decoder_embedding = decoder_embedding + pos_emb.unsqueeze(0)
        
        text = self.generator.generate(
            inputs_embeds=decoder_embedding,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.generator.config.eos_token_id,
            num_beams=k,
            repetition_penalty=2.0
        )

        # removed unwanted tokens and cut the text at the first dot
        sents = []
        for i in range(len(text)):
            sent = self.tokenizer.decode(text[i], skip_special_tokens=True)
            sent = sent.replace('\xa0', ' ')
            sents.append(sent)

        return sents
    

class Encoder(nn.Module):
    def __init__(self, encoder:str, device:str):
        super(Encoder, self).__init__()
        if encoder == CLIP: 
            self.feature_extractor, self.preprocess = clip.load("ViT-B/32", device=device)
            if device == 'mps':   
                self.feature_extractor.type(torch.float32)
            self.feature_extractor.load_state_dict(torch.load('data/models/clip.pth', map_location=device))
            self.encoder_type = CLIP
            self.out_dim = self.feature_extractor.visual.transformer.width
        
        elif encoder == REMOTE_CLIP:
            self.feature_extractor, self.preprocess = clip.load("ViT-B/32", device=device)
            self.feature_extractor.type(torch.float32)
            checkpoint_path = hf_hub_download("chendelong/RemoteCLIP", "RemoteCLIP-ViT-B-32.pt", cache_dir='checkpoints')
            self.feature_extractor.load_state_dict(torch.load(checkpoint_path, map_location=device))
            self.encoder_type = CLIP
            self.out_dim = self.feature_extractor.visual.transformer.width
            
        elif encoder == VGG:
            self.feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights=VGG16_Weights.IMAGENET1K_V1).features # KEEP ONLY THE FEATURE EXTRACTION BRANCH
            self.preprocess = VGG16_Weights.IMAGENET1K_V1.transforms()
            self.out_dim = 512
            self.encoder_type = VGG
        else:
            raise ValueError('Encoder must be either clip or vgg')
    
    def visual_clip(self, x: torch.Tensor):
        assert self.encoder_type == CLIP

        x = self.feature_extractor.visual.conv1(x.type(self.feature_extractor.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.feature_extractor.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.feature_extractor.visual.positional_embedding.to(x.dtype)
        x = self.feature_extractor.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.feature_extractor.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.feature_extractor.visual.ln_post(x)
        
        return x
        
    def forward(self, images:torch.Tensor):
        if self.encoder_type == VGG:
            decoder_embedding = self.feature_extractor(images)
            decoder_embedding = decoder_embedding.permute(0, 2, 3, 1)
            decoder_embedding = decoder_embedding.view(decoder_embedding.shape[0], -1, decoder_embedding.shape[-1])
            
        elif self.encoder_type == CLIP:
            decoder_embedding = self.visual_clip(images)
        
        return decoder_embedding

class Adapter_Layer(nn.Module):
    def __init__(self, encoder_dim:int, embedding_dim:int, dropout_prob:int=0):
        super(Adapter_Layer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(encoder_dim, embedding_dim),
            nn.Dropout(dropout_prob)
        )
        
    def forward(self, x:torch.Tensor):
        return self.layers(x)