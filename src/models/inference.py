#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
u4f7fu7528u4fddu5b58u7684u878du5408u6a21u578bu5bf9u6d4bu8bd5u6570u636eu8fdbu884cu771fu5047u65b0u95fbu5206u7c7b
u8fd9u4e2au811au672cu76f4u63a5u4f7fu7528u8badu7ec3u811au672cu4e2du7684u51fdu6570u6765u8fdbu884cu6a21u578bu8bc4u4f30
"""

import os
import sys
import torch
import argparse
import json
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import DistilBertTokenizer, CLIPTokenizer, CLIPProcessor

# u6dfbu52a0u9879u76eeu6839u76eeu5f55u5230u8defu5f84
sys.path.append('/Users/wujianxiang/Documents/GitHub/models')

# u5bfcu5165u8badu7ec3u811au672cu4e2du7684u51fdu6570
from fusionModels.train_fusion_model import (
    setup_mps_device, 
    MultiModalDataset, 
    evaluate_model,
    create_fusion_model
)
from testWordModel.TextProcessingModel import TextProcessingModel
from testPictureModel.ImageProcessingModel import ImageProcessingModel

def main(args):
    # u8bbeu7f6eu8bbeu5907
    device = setup_mps_device()
    print(f"u4f7fu7528u8bbeu5907: {device}")
    
    # u52a0u8f7du5206u8bcdu5668u548cu5904u7406u5668
    print("u52a0u8f7du9884u8badu7ec3u6a21u578bu548cu5904u7406u5668...")
    bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', local_files_only=True)
    clip_tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
    
    # u52a0u8f7dCLIPu5904u7406u5668
    try:
        # u9996u5148u5c1du8bd5u4eceu672cu5730u8defu5f84u52a0u8f7d
        clip_processor_path = os.path.join(args.model_cache_dir, 'clip-vit-base-patch32')
        if os.path.exists(clip_processor_path):
            print(f"u5c1du8bd5u4eceu672cu5730u8defu5f84u52a0u8f7dCLIPu5904u7406u5668: {clip_processor_path}")
            clip_processor = CLIPProcessor.from_pretrained(clip_processor_path, local_files_only=True)
            print("u6210u529fu4eceu672cu5730u8defu5f84u52a0u8f7dCLIPu5904u7406u5668")
        else:
            # u5982u679cu672cu5730u8defu5f84u4e0du5b58u5728uff0cu5c1du8bd5u4eceu9884u8badu7ec3u6a21u578bu52a0u8f7d
            print("u5c1du8bd5u4eceu9884u8badu7ec3u6a21u578bu52a0u8f7dCLIPu5904u7406u5668: openai/clip-vit-base-patch32")
            clip_processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', cache_dir=args.model_cache_dir)
            print("u6210u529fu4eceu9884u8badu7ec3u6a21u578bu52a0u8f7dCLIPu5904u7406u5668")
    except Exception as e:
        print(f"u52a0u8f7dCLIPu5904u7406u5668u5931u8d25: {e}")
        print("u521bu5efau9ed8u8ba4u7684CLIPu5904u7406u5668")
        # u4f7fu7528u9ed8u8ba4u7684u5904u7406u5668
        from transformers import CLIPImageProcessor
        clip_processor = CLIPProcessor(
            tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True),
            image_processor=CLIPImageProcessor.from_pretrained('openai/clip-vit-base-patch32', local_files_only=True)
        )
    
    # u52a0u8f7du6d4bu8bd5u6570u636e
    print("u52a0u8f7du6d4bu8bd5u6570u636e...")
    test_path = os.path.join(args.data_dir, args.test_file)
    test_images_dir = os.path.join(args.data_dir, args.test_images_dir)
    
    # u521bu5efau6d4bu8bd5u6570u636eu96c6
    test_dataset = MultiModalDataset(
        data_path=test_path,
        images_dir=test_images_dir,
        bert_tokenizer=bert_tokenizer,
        clip_tokenizer=clip_tokenizer,
        clip_processor=clip_processor,
        max_samples=args.max_samples
    )
    
    # u521bu5efau6570u636eu52a0u8f7du5668
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"u6d4bu8bd5u6570u636eu96c6u5927u5c0f: {len(test_dataset)} u6837u672c")
    
    # u52a0u8f7du6a21u578b
    print(f"u4ece {args.model_path} u52a0u8f7du6a21u578b...")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    # u521bu5efau6587u672cu548cu56feu50cfu6a21u578b
    print("u521bu5efau6587u672cu548cu56feu50cfu6a21u578b...")
    # u521bu5efau6587u672cu6a21u578b
    text_model = TextProcessingModel(
        bert_model_path='distilbert-base-uncased',
        clip_model_path='openai/clip-vit-base-patch32',
        use_local_models=True
    )
    
    # u521bu5efau56feu50cfu6a21u578b
    image_model = ImageProcessingModel(
        clip_model_path='openai/clip-vit-base-patch32',
        use_local_models=True,
        fast_mode=True
    )
    
    # u8bbeu7f6eu4e3au8bc4u4f30u6a21u5f0f
    text_model.eval()
    image_model.eval()
    
    # u79fbu52a8u6a21u578bu5230u6307u5b9au8bbeu5907
    text_model.to(device)
    image_model.to(device)
    
    # u521bu5efau878du5408u6a21u578b
    print("u521bu5efau878du5408u6a21u578b...")
    fusion_model = create_fusion_model(
        text_model=text_model,
        image_model=image_model,
        fusion_dim=args.fusion_dim,
        num_classes=args.num_classes
    )
    
    # u52a0u8f7du6a21u578bu72b6u6001u5b57u5178
    print("u52a0u8f7du6a21u578bu72b6u6001u5b57u5178...")
    model_path = args.model_path
    try:
        # u4f7fu7528strict=Falseu52a0u8f7du6a21u578bu72b6u6001u5b57u5178uff0cu5141u8bb8u8df3u8fc7u4e0du5339u914du7684u53c2u6570
        state_dict = torch.load(model_path, map_location=device)
        # u5982u679cu662fu5b8cu6574u7684u68c0u67e5u70b9uff0cu63d0u53d6model_state_dict
        if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        
        # u5c1du8bd5u76f4u63a5u52a0u8f7du6a21u578bu72b6u6001u5b57u5178
        fusion_model.load_state_dict(state_dict, strict=False)
        print("u6a21u578bu72b6u6001u5b57u5178u52a0u8f7du6210u529f (u975eu4e25u683cu6a21u5f0f)")
    except Exception as e:
        print(f"u52a0u8f7du6a21u578bu72b6u6001u5b57u5178u65f6u51fau9519: {e}")
        print("u5c1du8bd5u624bu52a8u5339u914du53c2u6570...")
        
        # u6253u5370u6a21u578bu7ed3u6784u4fe1u606f
        print("\nu5f53u524du6a21u578bu7ed3u6784:")
        for name, param in fusion_model.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.data.shape}")
        
        try:
            # u624bu52a8u5339u914du53c2u6570
            state_dict = torch.load(model_path, map_location=device)
            if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            # u53eau52a0u8f7du5f62u72b6u5339u914du7684u53c2u6570
            model_dict = fusion_model.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            fusion_model.load_state_dict(model_dict, strict=False)
            print(f"u90e8u5206u52a0u8f7du6a21u578bu72b6u6001u5b57u5178uff0cu6210u529fu52a0u8f7d {len(pretrained_dict)}/{len(model_dict)} u4e2au53c2u6570")
            
            # u6253u5370u672au52a0u8f7du7684u53c2u6570
            missing_keys = [k for k in model_dict.keys() if k not in pretrained_dict]
            if len(missing_keys) > 0:
                print("\nu672au52a0u8f7du7684u53c2u6570:")
                for key in missing_keys[:10]:  # u53eau6253u5370u524d10u4e2a
                    print(f"{key}: {model_dict[key].shape}")
                if len(missing_keys) > 10:
                    print(f"...u8fd8u6709 {len(missing_keys) - 10} u4e2au672au663eu793a")
        except Exception as e:
            print(f"u624bu52a8u5339u914du53c2u6570u65f6u51fau9519: {e}")
    
    fusion_model.to(device)
    fusion_model.eval()
    
    # u8bc4u4f30u6a21u578b
    print("u5f00u59cbu8bc4u4f30u6a21u578b...")
    metrics = evaluate_model(
        model=fusion_model,
        test_loader=test_loader,
        device=device
    )
    
    # u81eau5b9au4e49u51fdu6570u6536u96c6u9884u6d4bu7ed3u679cu548cu6807u7b7e
    def collect_predictions(model, data_loader, device):
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="u6536u96c6u9884u6d4bu7ed3u679c"):
                try:
                    # u5c06u6570u636eu79fbu52a8u5230u8bbeu5907
                    bert_input_ids = batch['bert_input_ids'].to(device)
                    bert_attention_mask = batch['bert_attention_mask'].to(device)
                    clip_input_ids = batch['clip_input_ids'].to(device)
                    clip_attention_mask = batch['clip_attention_mask'].to(device)
                    resnet_image = batch['resnet_image'].to(device)
                    clip_pixel_values = batch['clip_pixel_values'].to(device)
                    labels = batch['label'].to(device)
                    
                    # u524du5411u4f20u64ad
                    outputs = model(
                        bert_input_ids=bert_input_ids,
                        bert_attention_mask=bert_attention_mask,
                        clip_input_ids=clip_input_ids,
                        clip_attention_mask=clip_attention_mask,
                        resnet_image=resnet_image,
                        clip_pixel_values=clip_pixel_values
                    )
                    
                    # u5982u679cu8f93u51fau662fu5143u7ec4uff0cu53d6u7b2cu4e00u4e2au5143u7d20uff08logitsuff09
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                    
                    # u83b7u53d6u9884u6d4b
                    _, preds = torch.max(logits, dim=1)
                    
                    # u6536u96c6u9884u6d4bu548cu6807u7b7e
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                
                except Exception as e:
                    print(f"u6536u96c6u9884u6d4bu51fau9519: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
        
        return all_preds, all_labels
    
    # u6536u96c6u9884u6d4bu7ed3u679cu548cu6807u7b7e
    print("u6536u96c6u8be6u7ec6u9884u6d4bu7ed3u679c...")
    all_preds, all_labels = collect_predictions(fusion_model, test_loader, device)
    
    # u663eu793au7ed3u679c
    print("\nu8bc4u4f30u7ed3u679c:")
    print(f"u51c6u786eu7387: {accuracy_score(all_labels, all_preds):.4f}")
    print(f"u7cbeu786eu7387: {precision_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"u53ecu56deu7387: {recall_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print(f"F1u503c: {f1_score(all_labels, all_preds, average='weighted', zero_division=0):.4f}")
    print("\nu6df7u6dc6u77e9u9635:")
    print(confusion_matrix(all_labels, all_preds))
    
    # u7ed8u5236u6df7u6dc6u77e9u9635
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(all_labels, all_preds), annot=True, fmt='d', cmap='Blues')
    plt.xlabel('u9884u6d4bu6807u7b7e')
    plt.ylabel('u771fu5b9eu6807u7b7e')
    plt.title('u6df7u6dc6u77e9u9635')
    plt.savefig('evaluation_confusion_matrix.png')
    print(f"u6df7u6dc6u77e9u9635u56feu50cfu5df2u4fddu5b58u5230: {os.path.abspath('evaluation_confusion_matrix.png')}")
    
    # u4fddu5b58u8bc4u4f30u7ed3u679c
    results = {
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'f1': float(f1_score(all_labels, all_preds, average='weighted', zero_division=0)),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
        'predictions': {
            'true_positives': int(np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 1))),
            'false_positives': int(np.sum((np.array(all_preds) == 1) & (np.array(all_labels) == 0))),
            'true_negatives': int(np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 0))),
            'false_negatives': int(np.sum((np.array(all_preds) == 0) & (np.array(all_labels) == 1)))
        }
    }
    
    with open('prediction_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"u9884u6d4bu7ed3u679cu5df2u4fddu5b58u5230: {os.path.abspath('prediction_results.json')}")
    
    # u4fddu5b58u8be6u7ec6u7684u9884u6d4bu7ed3u679c
    detailed_results = []
    for i, (pred, label) in enumerate(zip(all_preds, all_labels)):
        if i < len(test_dataset.data):
            sample = test_dataset.data.iloc[i]
            detailed_results.append({
                'text': sample['text'],
                'image_path': sample['path'],
                'true_label': int(label),
                'predicted_label': int(pred),
                'correct': int(label) == int(pred)
            })
    
    with open('detailed_predictions.json', 'w') as f:
        json.dump(detailed_results, f, indent=4)
    print(f"u8be6u7ec6u9884u6d4bu7ed3u679cu5df2u4fddu5b58u5230: {os.path.abspath('detailed_predictions.json')}")

if __name__ == '__main__':
    # u89e3u6790u547du4ee4u884cu53c2u6570
    parser = argparse.ArgumentParser(description='u4f7fu7528u4fddu5b58u7684u878du5408u6a21u578bu5bf9u6d4bu8bd5u6570u636eu8fdbu884cu771fu5047u65b0u95fbu5206u7c7b')
    
    # u6570u636eu53c2u6570
    parser.add_argument('--data_dir', type=str, default='/Users/wujianxiang/Documents/GitHub/models/data',
                        help='u6570u636eu76eeu5f55')
    parser.add_argument('--test_file', type=str, default='test.csv',
                        help='u6d4bu8bd5u6570u636eu6587u4ef6u540d')
    parser.add_argument('--test_images_dir', type=str, default='images',
                        help='u6d4bu8bd5u56feu50cfu76eeu5f55u540d')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='u6700u5927u6837u672cu6570uff0cu7528u4e8eu8c03u8bd5')
    
    # u6a21u578bu53c2u6570
    parser.add_argument('--model_path', type=str, default='/Users/wujianxiang/Documents/GitHub/models/saved_models/fusion_model.pth',
                        help='u878du5408u6a21u578bu8defu5f84')
    parser.add_argument('--fusion_dim', type=int, default=512,
                        help='u878du5408u7279u5f81u7ef4u5ea6')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='u5206u7c7bu7c7bu522bu6570')
    
    # u5176u4ed6u53c2u6570
    parser.add_argument('--batch_size', type=int, default=16,
                        help='u6279u6b21u5927u5c0f')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='u6570u636eu52a0u8f7du5668u5de5u4f5cu8fdbu7a0bu6570')
    parser.add_argument('--model_cache_dir', type=str, default='/Users/wujianxiang/Documents/GitHub/models/model_cache',
                        help='u6a21u578bu7f13u5b58u76eeu5f55')
    
    args = parser.parse_args()
    main(args)
