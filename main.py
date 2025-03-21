

#explain to old person that there is an attack

#fine-tune VisualbERT on UFC crime data to create a crime detecting visual to text chatbot
#use UCF crime dataset to train the model https://www.kaggle.com/datasets/odins0n/ucf-crime-dataset/data

# use the output of the vido description to do the classification.

# use the camera to record and analyze the video

import torch
from transformers import BertTokenizer, VisualBertModel

model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
tokenizer = BertTokenizer.from_pretrained("google-bert/bert-base-uncased")

inputs = tokenizer("What is the man eating?", return_tensors="pt")
# this is a custom function that returns the visual embeddings given the image path
visual_embeds = get_visual_embeddings('ucf_crime_dataset')

visual_token_type_ids = torch.ones(visual_embeds.shape[:-1], dtype=torch.long)
visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
inputs.update(
    {
        "visual_embeds": visual_embeds,
        "visual_token_type_ids": visual_token_type_ids,
        "visual_attention_mask": visual_attention_mask,
    }
)
outputs = model(**inputs)
last_hidden_state = outputs.last_hidden_state

