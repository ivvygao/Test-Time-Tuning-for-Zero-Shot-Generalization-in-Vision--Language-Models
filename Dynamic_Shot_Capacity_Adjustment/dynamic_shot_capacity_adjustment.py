
####源码
# import random
# import argparse
# import wandb
# from tqdm import tqdm
# from datetime import datetime
# from collections import defaultdict
#
# import torch
# import torch.nn.functional as F
# import operator
#
# import clip
# from utils import *
#
#
# def get_arguments():
#     """Get arguments of the test-time adaptation."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', dest='config', required=True,
#                         help='settings of TDA on specific dataset in yaml format.')
#     parser.add_argument('--wandb-log', dest='wandb', action='store_true',
#                         help='Whether you want to log to wandb. Include this flag to enable logging.')
#     parser.add_argument('--datasets', dest='datasets', type=str, required=True,
#                         help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
#     parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/',
#                         help='Path to the datasets directory. Default is ./dataset/')
#     parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True,
#                         help='CLIP model backbone to use: RN50 or ViT-B/16.')
#
#     args = parser.parse_args()
#
#     return args
#
#
# def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
#     """Update cache with new features and loss, maintaining the maximum shot capacity."""
#     with torch.no_grad():
#         item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
#         if pred in cache:
#             if len(cache[pred]) < shot_capacity:
#                 cache[pred].append(item)
#             elif features_loss[1] < cache[pred][-1][1]:
#                 cache[pred][-1] = item
#             cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
#         else:
#             cache[pred] = [item]
#
#
# def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
#     """Compute logits using positive/negative cache."""
#     with torch.no_grad():
#         cache_keys = []
#         cache_values = []
#         for class_index in sorted(cache.keys()):
#             for item in cache[class_index]:
#                 cache_keys.append(item[0])
#                 if neg_mask_thresholds:
#                     cache_values.append(item[2])
#                 else:
#                     cache_values.append(class_index)
#
#         cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
#         if neg_mask_thresholds:
#             cache_values = torch.cat(cache_values, dim=0)
#             cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(
#                 torch.int8)).cuda().half()
#         else:
#             cache_values = (
#                 F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()
#
#         affinity = image_features @ cache_keys
#         cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
#         return alpha * cache_logits
#
#
# def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights):
#     with torch.no_grad():
#         pos_cache, neg_cache, accuracies = {}, {}, []
#
#         # Unpack all hyperparameters
#         pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
#         if pos_enabled:
#             pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
#         if neg_enabled:
#             neg_params = {k: neg_cfg[k] for k in
#                           ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}
#
#         # Test-time adaptation
#         for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
#             image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
#             target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)
#
#             if pos_enabled:
#                 update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])
#
#             if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < \
#                     neg_params['entropy_threshold']['upper']:
#                 update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)
#
#             final_logits = clip_logits.clone()
#             if pos_enabled and pos_cache:
#                 final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'],
#                                                      clip_weights)
#             if neg_enabled and neg_cache:
#                 final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'],
#                                                      clip_weights, (neg_params['mask_threshold']['lower'],
#                                                                     neg_params['mask_threshold']['upper']))
#
#             acc = cls_acc(final_logits, target)
#             accuracies.append(acc)
#             wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)
#
#             if i % 1000 == 0:
#                 print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))
#         print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))
#         return sum(accuracies) / len(accuracies)
#
#
# def main():
#     args = get_arguments()
#     config_path = args.config
#
#     # Initialize CLIP model
#     clip_model, preprocess = clip.load(args.backbone)
#     clip_model.eval()
#
#     # Set random seed
#     random.seed(1)
#     torch.manual_seed(1)
#
#     if args.wandb:
#         date = datetime.now().strftime("%b%d_%H-%M-%S")
#         group_name = f"{args.backbone}_{args.datasets}_{date}"
#
#     # Run TDA on each dataset
#     datasets = args.datasets.split('/')
#     for dataset_name in datasets:
#         print(f"Processing {dataset_name} dataset.")
#
#         cfg = get_config_file(config_path, dataset_name)
#         print("\nRunning dataset configurations:")
#         print(cfg, "\n")
#
#         test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
#         clip_weights = clip_classifier(classnames, template, clip_model)
#
#         if args.wandb:
#             run_name = f"{dataset_name}"
#             run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)
#
#         acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)
#
#         if args.wandb:
#             wandb.log({f"{dataset_name}": acc})
#             run.finish()
#
#
# if __name__ == "__main__":
#     main()



########################################################################################
# #
# import json
# import random
# import argparse
# from collections import defaultdict
#
# import wandb
# from tqdm import tqdm
# from datetime import datetime
#
# import torch
# import torch.nn.functional as F
# import operator
#
# import clip
# from utils import *
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
#
#
# def get_arguments():
#     """Get arguments of the test-time adaptation."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', dest='config', required=True,
#                         help='settings of TDA on specific dataset in yaml format.')
#     parser.add_argument('--wandb-log', dest='wandb', action='store_true',
#                         help='Whether you want to log to wandb. Include this flag to enable logging.')
#     parser.add_argument('--datasets', dest='datasets', type=str, required=True,
#                         help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
#     parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/',
#                         help='Path to the datasets directory. Default is ./dataset/')
#     parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True,
#                         help='CLIP model backbone to use: RN50 or ViT-B/16.')
#
#     args = parser.parse_args()
#
#     return args
#
#
# def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
#     """Update cache with new features and loss, maintaining the maximum shot capacity.
#     这个函数的目的是保持每个类别的缓存条目数量不超过 shot_capacity，并且优先保留损失最小的条目。
#     缓存更新逻辑：
# 	如果 pred 已经在 cache 中，且该类别的缓存条目数少于 shot capacity，那么直接将新条目 item 添加到缓存中。
# 	如果 cache[pred] 的条目数已达上限，那么就会根据 features_loss（损失值）来决定是否替换掉缓存中损失最大的条目（即质量最差的条目）。
# 	最终的 cache[pred] 会按照损失值从小到大排序，以确保缓存中始终保留最优质的预测。"""
#     # 使用 torch.no_grad() 上下文管理器，禁用梯度计算以节省内存和计算资源。
#     with torch.no_grad():
#         # 根据 include_prob_map 标志，决定是否包括概率映射（probability map）。
#         # 如果不包括，只保留 features_loss，否则包括前三项（图像特征、损失、概率映射）
#         item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
#         # pred 是模型对图像的预测类别，也就是模型认为图像属于哪个类的输出
#         if pred in cache:
#             if len(cache[pred]) < shot_capacity:
#                 cache[pred].append(item)
#             elif features_loss[1] < cache[pred][-1][1]:
#                 cache[pred][-1] = item
#             cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
#         else:
#             cache[pred] = [item]
#
#
# def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
#     """Compute logits using positive/negative cache."""
#
#     with torch.no_grad():
#         cache_keys = []
#         cache_values = []
#         for class_index in sorted(cache.keys()):
#             for item in cache[class_index]:
#                 cache_keys.append(item[0])
#                 if neg_mask_thresholds:
#                     cache_values.append(item[2])
#                 else:
#                     cache_values.append(class_index)
#
#         cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
#         if neg_mask_thresholds:
#             cache_values = torch.cat(cache_values, dim=0)
#             cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(
#                 torch.int8)).cuda().half()
#         else:
#             cache_values = (
#                 F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()
#
#         affinity = image_features @ cache_keys
#         cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
#         return alpha * cache_logits
#
#
# def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, pos_shot_capacity_dict, neg_shot_capacity, output_file="class_accuracies.json"):
#     with torch.no_grad():
#         pos_cache, neg_cache, accuracies = {}, {}, []
#         top_accuracies, all_preds, all_targets = [],[],[]
#
#
#         ###############
#         class_accuracies = defaultdict(list)
#         # shot_capacity_dict = defaultdict(lambda:3)
#         ###############
#
#         # Unpack all hyperparameters
#         pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
#         if pos_enabled:
#             pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
#         if neg_enabled:
#             neg_params = {k: neg_cfg[k] for k in
#                           ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}
#
#         # Test-time adaptation
#         # 【在这行代码中，loader 每次返回一个批次的图像 images 和对应的标签 target。
#         # target 是一个张量，包含了当前批次中每个图像的真实类别标签。这些标签是整数，代表数据集中每个图像所属的类别。
#         for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
#             # 计算图像特征、模型输出的 logits、损失、概率映射和预测。
#             image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
#             target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)
#
#
#             # 根据模型预测和损失更新正缓存和负缓存，保持最大容量。
#             if pos_enabled:
#
#                 #######################
#                 update_cache(pos_cache, pred, [image_features, loss], pos_shot_capacity_dict.get(pred, 3))
#                 ######################
#                 #update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])
#
#             if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < \
#                     neg_params['entropy_threshold']['upper']:
#                 update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_shot_capacity, True)
#                 #update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)
#
#             # 调整最终 logits
#             # 作用: 使用正负缓存调整模型输出的 logits。
#             final_logits = clip_logits.clone()
#             if pos_enabled and pos_cache:
#                 final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'],
#                                                      clip_weights)
#             if neg_enabled and neg_cache:
#                 final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'],
#                                                      clip_weights, (neg_params['mask_threshold']['lower'],
#                                                                     neg_params['mask_threshold']['upper']))
#
#             # 计算分类准确率。
#             acc = cls_acc(final_logits, target)
#             accuracies.append(acc)
#
#             ############
#             _, topk_preds = final_logits.topk(3, dim=1)
#             correct_topk = topk_preds.eq(target.view(-1, 1).expand_as(topk_preds))
#             top_accuracy = correct_topk.any(dim=1).float().mean().item()
#             top_accuracies.append(top_accuracy)
#
#             all_preds.extend(topk_preds[:, 0].cpu().numpy())
#             all_targets.extend(target.cpu().numpy())
#
#
#             ############
#
#             #################保存每个类别的准确率###############
#             for idx, label in enumerate(target):
#                 class_accuracies[label.item()].append(final_logits[idx].max(0)[1].eq(label).item())
#             if (i+1) % 200 ==0:
#                 adjust_shot_capacity(class_accuracies, pos_shot_capacity_dict)
#                 class_accuracies.clear()
#                 print(f"Adjusted shot capacity at step {i + 1}")
#
#                 precision, recall, fscore, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro',
#                                                                                zero_division=0)
#                 micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(all_targets, all_preds,
#                                                                                                  average='micro',
#                                                                                                  zero_division=0)
#                 weighted_precision, weighted_recall, weighted_fscore, _ = precision_recall_fscore_support(all_targets,
#                                                                                                           all_preds,
#                                                                                                           average='weighted',
#                                                                                                           zero_division=0)
#
#                 print(f"---- Processed {i + 1} images ----")
#                 avg_top3_acc = sum(top_accuracies) / len(top_accuracies)
#                 print(f"Current Top-3 Accuracy: {avg_top3_acc:.2f}%")
#                 print(f"Macro Precision: {precision:.3f}, Macro Recall: {recall:.3f}, Macro F1-Score: {fscore:.3f}")
#                 print(
#                     f"Micro Precision: {micro_precision:.3f}, Micro Recall: {micro_recall:.3f}, Micro F1-Score: {micro_fscore:.3f}")
#                 print(
#                     f"Weighted Precision: {weighted_precision:.3f}, Weighted Recall: {weighted_recall:.3f}, Weighted F1-Score: {weighted_fscore:.3f}")
#
#             ########################
#             # if wandb:
#             wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)
#
#             if i % 1000 == 0:
#                 print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))
#
#         ########################################################
#         avg_class_accuracies = {k: sum(v) / len(v) for k, v in class_accuracies.items()}
#         # 输出到文件
#         with open(output_file, 'w') as f:
#             json.dump(avg_class_accuracies, f, indent=4)  # 保存为JSON文件
#         #######################################
#
#         print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))
#
#         ############3# 输出评估结果
#         print(f"Currrent Positive Shot Capacity:{pos_shot_capacity_dict}")
#         print(f"Top-3 Accuracy: {np.mean(top_accuracies):.2f}%")
#         print(f"Macro Precision: {precision:.3f}, Macro Recall: {recall:.3f}, Macro F1-Score: {fscore:.3f}")
#         print(
#             f"Micro Precision: {micro_precision:.3f}, Micro Recall: {micro_recall:.3f}, Micro F1-Score: {micro_fscore:.3f}")
#         print(
#             f"Weighted Precision: {weighted_precision:.3f}, Weighted Recall: {weighted_recall:.3f}, Weighted F1-Score: {weighted_fscore:.3f}")
#         ##########################
#
#         return sum(accuracies) / len(accuracies), class_accuracies  # 这里新加了一个class的输出
#
#
# ###########################################改进一些代码########################################################
# def adjust_shot_capacity(class_accuracies, shot_capacity_dict, lower_threshold=0.3, upper_threshold=0.8,
#                          max_capacity=15, min_capacity=2):
#     """
#     根据类别的准确率动态调整 Shot Capacity。
#
#     :param class_accuracies: 字典，包含每个类别的准确率。
#     :param shot_capacity_dict: 字典，存储每个类别的 Shot Capacity。
#     :param lower_threshold: 低于该准确率时增加 Shot Capacity。
#     :param upper_threshold: 高于该准确率时减少 Shot Capacity。
#     :param max_capacity: 最大 Shot Capacity。
#     :param min_capacity: 最小 Shot Capacity。
#     """
#     changes = {}  # 记录变化的类别和其容量变化
#     for label, correct_list in class_accuracies.items():
#         if len(correct_list) > 0:  # 确保该类别在当前批次中有涉及到
#             accuracy = sum(correct_list) / len(correct_list)
#             original_capacity = shot_capacity_dict[label]
#             if accuracy < lower_threshold:  # 如果准确率低于下限，增加 shot capacity
#                 shot_capacity_dict[label] = min(shot_capacity_dict[label] + 1, max_capacity)
#             elif accuracy > upper_threshold:  # 如果准确率高于上限，减少 shot capacity
#                 shot_capacity_dict[label] = max(shot_capacity_dict[label] - 1, min_capacity)
#
#                 # 如果容量发生变化，则记录
#             if shot_capacity_dict[label] != original_capacity:
#                 changes[label] = shot_capacity_dict[label]
#
#
#     # 如果有变化，打印并记录到 WandB
#     if changes:
#         print(f"Shot capacity adjustments made:")
#         for label, new_capacity in changes.items():
#             print(f"Class {label}: {shot_capacity_dict[label]}")
#
#
#         #wandb.log({"shot_capacity_changes": changes})
#
#
# from sklearn.metrics import classification_report
#
# def print_classification_report(predictions, targets, class_names):
#     """
#     打印分类报告。
#     :param predictions: 预测列表。
#     :param targets: 实际目标列表。
#     :param class_names: 类名列表。
#     """
#     report = classification_report(targets, predictions, target_names=class_names)
#     print(report)
#
#
# #############################################################################################################
# # def main():
# # args = get_arguments()
# # config_path = args.config
# #
# # # Initialize CLIP model
# # clip_model, preprocess = clip.load(args.backbone)
# # clip_model.eval()
# #
# # # Set random seed
# # random.seed(1)
# # torch.manual_seed(1)
# #
# # if args.wandb:
# #     date = datetime.now().strftime("%b%d_%H-%M-%S")
# #     group_name = f"{args.backbone}_{args.datasets}_{date}"
# #
# # # Run TDA on each dataset
# # datasets = args.datasets.split('/')
# # for dataset_name in datasets:
# #     print(f"Processing {dataset_name} dataset.")
# #
# #     cfg = get_config_file(config_path, dataset_name)
# #     print("\nRunning dataset configurations:")
# #     print(cfg, "\n")
# #
# #     test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
# #     clip_weights = clip_classifier(classnames, template, clip_model)
# #
# #     if args.wandb:
# #         run_name = f"{dataset_name}"
# #         run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)
# #
# #     acc = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights)
# #
# #     if args.wandb:
# #         wandb.log({f"{dataset_name}": acc})
# #         run.finish()
#
# def main():
#     args = get_arguments()
#     config_path = args.config
#
#     # Initialize CLIP model
#     clip_model, preprocess = clip.load(args.backbone)
#     clip_model.eval()
#
#     # Set random seed
#     random.seed(1)
#     torch.manual_seed(1)
#
#     if args.wandb:
#         date = datetime.now().strftime("%b%d_%H-%M-%S")
#         group_name = f"{args.backbone}_{args.datasets}_{date}"
#
#     # Run TDA on each dataset
#     datasets = args.datasets.split('/')
#     for dataset_name in datasets:
#         print(f"Processing {dataset_name} dataset.")
#
#         cfg = get_config_file(config_path, dataset_name)
#         print("\nRunning dataset configurations:")
#         print(cfg, "\n")
#
#         test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
#         clip_weights = clip_classifier(classnames, template, clip_model)
#
#         # Initialize the shot capacity dict for positive cache
#         pos_shot_capacity_dict = {label: cfg['positive']['shot_capacity'] for label in range(len(classnames))}
#         neg_shot_capacity = cfg['negative']['shot_capacity']  # Negative cache capacity remains constant
#
#         if args.wandb:
#             run_name = f"{dataset_name}"
#             run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)
#
#         acc, class_accuracires = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights,
#                                               pos_shot_capacity_dict, neg_shot_capacity, output_file="class_accuracies.json")
#
#         if args.wandb:
#             wandb.log({f"{dataset_name}": acc})
#             run.finish()
#
#
#
#
# if __name__ == "__main__":
#     main()

##################################################
#
import json
import random
import argparse
from collections import defaultdict

import wandb
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import operator

import clip
from utils import *
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support


def get_arguments():
    """Get arguments of the test-time adaptation."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', dest='config', required=True,
                        help='settings of TDA on specific dataset in yaml format.')
    parser.add_argument('--wandb-log', dest='wandb', action='store_true',
                        help='Whether you want to log to wandb. Include this flag to enable logging.')
    parser.add_argument('--datasets', dest='datasets', type=str, required=True,
                        help="Datasets to process, separated by a slash (/). Example: I/A/V/R/S")
    parser.add_argument('--data-root', dest='data_root', type=str, default='./dataset/',
                        help='Path to the datasets directory. Default is ./dataset/')
    parser.add_argument('--backbone', dest='backbone', type=str, choices=['RN50', 'ViT-B/16'], required=True,
                        help='CLIP model backbone to use: RN50 or ViT-B/16.')

    args = parser.parse_args()

    return args


def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache with new features and loss, maintaining the maximum shot capacity.
    这个函数的目的是保持每个类别的缓存条目数量不超过 shot_capacity，并且优先保留损失最小的条目。
    缓存更新逻辑：
	如果 pred 已经在 cache 中，且该类别的缓存条目数少于 shot capacity，那么直接将新条目 item 添加到缓存中。
	如果 cache[pred] 的条目数已达上限，那么就会根据 features_loss（损失值）来决定是否替换掉缓存中损失最大的条目（即质量最差的条目）。
	最终的 cache[pred] 会按照损失值从小到大排序，以确保缓存中始终保留最优质的预测。"""
    # 使用 torch.no_grad() 上下文管理器，禁用梯度计算以节省内存和计算资源。
    with torch.no_grad():
        # 根据 include_prob_map 标志，决定是否包括概率映射（probability map）。
        # 如果不包括，只保留 features_loss，否则包括前三项（图像特征、损失、概率映射）
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        # pred 是模型对图像的预测类别，也就是模型认为图像属于哪个类的输出
        if pred in cache:
            if len(cache[pred]) < shot_capacity:
                cache[pred].append(item)
            elif features_loss[1] < cache[pred][-1][1]:
                cache[pred][-1] = item
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            cache[pred] = [item]


def compute_cache_logits(image_features, cache, alpha, beta, clip_weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache."""

    with torch.no_grad():
        cache_keys = []
        cache_values = []
        for class_index in sorted(cache.keys()):
            for item in cache[class_index]:
                cache_keys.append(item[0])
                if neg_mask_thresholds:
                    cache_values.append(item[2])
                else:
                    cache_values.append(class_index)

        cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
        if neg_mask_thresholds:
            cache_values = torch.cat(cache_values, dim=0)
            cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(
                torch.int8)).cuda().half()
        else:
            cache_values = (
                F.one_hot(torch.Tensor(cache_values).to(torch.int64), num_classes=clip_weights.size(1))).cuda().half()

        affinity = image_features @ cache_keys
        cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
        return alpha * cache_logits


#def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, pos_shot_capacity_dict, neg_shot_capacity, output_dir="outputs"):
def run_test_tda(pos_cfg, neg_cfg, loader, clip_model, clip_weights, pos_shot_capacity_dict, neg_shot_capacity,
                     output_dir="outputs"):

    with torch.no_grad():
        pos_cache, neg_cache, accuracies = {}, {}, []
        top_accuracies, all_preds, all_targets = [],[],[]


        ###############
        class_probabilities = defaultdict(list)  # 存储预测概率
        class_labels = defaultdict(list)         # 存储实际标签
        class_accuracies = defaultdict(list)     # 存储准确率
        class_accuracies_output = defaultdict(list)
        misclassifications = []                  # 错误分类数据
        shot_capacity_changes = defaultdict(list) # shot capacity 变化
        predicted_labels = defaultdict(list)     # 存储每个类别的预测结果

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ###############

        # Unpack all hyperparameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        if pos_enabled:
            pos_params = {k: pos_cfg[k] for k in ['shot_capacity', 'alpha', 'beta']}
        if neg_enabled:
            neg_params = {k: neg_cfg[k] for k in
                          ['shot_capacity', 'alpha', 'beta', 'entropy_threshold', 'mask_threshold']}

        # Test-time adaptation
        # 【在这行代码中，loader 每次返回一个批次的图像 images 和对应的标签 target。
        # target 是一个张量，包含了当前批次中每个图像的真实类别标签。这些标签是整数，代表数据集中每个图像所属的类别。
        for i, (images, target) in enumerate(tqdm(loader, desc='Processed test images: ')):
            # 计算图像特征、模型输出的 logits、损失、概率映射和预测。
            image_features, clip_logits, loss, prob_map, pred = get_clip_logits(images, clip_model, clip_weights)
            target, prop_entropy = target.cuda(), get_entropy(loss, clip_weights)


            # 根据模型预测和损失更新正缓存和负缓存，保持最大容量。
            if pos_enabled:

                #######################
                update_cache(pos_cache, pred, [image_features, loss], pos_shot_capacity_dict.get(pred, 3))
                ######################
                #update_cache(pos_cache, pred, [image_features, loss], pos_params['shot_capacity'])

            if neg_enabled and neg_params['entropy_threshold']['lower'] < prop_entropy < \
                    neg_params['entropy_threshold']['upper']:
                update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_shot_capacity, True)
                #update_cache(neg_cache, pred, [image_features, loss, prob_map], neg_params['shot_capacity'], True)

            # 调整最终 logits
            # 作用: 使用正负缓存调整模型输出的 logits。
            final_logits = clip_logits.clone()
            if pos_enabled and pos_cache:
                final_logits += compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'],
                                                     clip_weights)
            if neg_enabled and neg_cache:
                final_logits -= compute_cache_logits(image_features, neg_cache, neg_params['alpha'], neg_params['beta'],
                                                     clip_weights, (neg_params['mask_threshold']['lower'],
                                                                    neg_params['mask_threshold']['upper']))



            # 计算分类准确率。
            acc = cls_acc(final_logits, target)
            accuracies.append(acc)

            ############
            _, topk_preds = final_logits.topk(3, dim=1)
            correct_topk = topk_preds.eq(target.view(-1, 1).expand_as(topk_preds))
            top_accuracy = correct_topk.any(dim=1).float().mean().item()
            top_accuracies.append(top_accuracy)

            all_preds.extend(topk_preds[:, 0].cpu().numpy())
            all_targets.extend(target.cpu().numpy())



            ############
            probs = torch.softmax(final_logits, dim=1)
            #################保存每个类别的准确率###############
            for idx, label in enumerate(target):
                class_accuracies[label.item()].append(final_logits[idx].max(0)[1].eq(label).item())
                class_labels[label.item()].append(label.item())
                class_probabilities[label.item()].append(probs[idx].cpu().numpy().tolist())
                is_correct = probs[idx].argmax().item() == label.item()
                predicted_labels[label.item()].append(probs[idx].argmax().item())
                class_accuracies_output[label.item()].append(is_correct)

                if not is_correct:
                    misclassifications.append({'image_idx': i, 'predicted': probs[idx].argmax().item(), 'true': label.item()})

            if (i+1) % 200 ==0:
                adjust_shot_capacity(class_accuracies, pos_shot_capacity_dict)
                for label, capacity in pos_shot_capacity_dict.items():
                    shot_capacity_changes[label].append(capacity)

                class_accuracies.clear()
                print(f"Adjusted shot capacity at step {i + 1}")

                precision, recall, fscore, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro',
                                                                               zero_division=0)
                micro_precision, micro_recall, micro_fscore, _ = precision_recall_fscore_support(all_targets, all_preds,
                                                                                                 average='micro',
                                                                                                 zero_division=0)
                weighted_precision, weighted_recall, weighted_fscore, _ = precision_recall_fscore_support(all_targets,
                                                                                                          all_preds,
                                                                                                          average='weighted',
                                                                                                          zero_division=0)

                print(f"---- Processed {i + 1} images ----")
                avg_top3_acc = sum(top_accuracies) / len(top_accuracies)
                print(f"Current Top-3 Accuracy: {avg_top3_acc:.2f}%")
                print(f"Macro Precision: {precision:.3f}, Macro Recall: {recall:.3f}, Macro F1-Score: {fscore:.3f}")
                print(
                    f"Micro Precision: {micro_precision:.3f}, Micro Recall: {micro_recall:.3f}, Micro F1-Score: {micro_fscore:.3f}")
                print(
                    f"Weighted Precision: {weighted_precision:.3f}, Weighted Recall: {weighted_recall:.3f}, Weighted F1-Score: {weighted_fscore:.3f}")

            ########################
            # if wandb:
            wandb.log({"Averaged test accuracy": sum(accuracies) / len(accuracies)}, commit=True)

            if i % 1000 == 0:
                print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

        ########################################################
        avg_class_accuracies = {k: sum(v) / len(v) for k, v in class_accuracies.items()}
        # 输出到文件
        # 保存数据
        all_preds = [int(pred) for pred in all_preds]
        all_targets = [int(target) for target in all_targets]



        #######################################

        print("---- TDA's test accuracy: {:.2f}. ----\n".format(sum(accuracies) / len(accuracies)))

        ############3# 输出评估结果
        print(f"Currrent Positive Shot Capacity:{pos_shot_capacity_dict}")
        print(f"Top-3 Accuracy: {np.mean(top_accuracies):.2f}%")
        print(f"Macro Precision: {precision:.3f}, Macro Recall: {recall:.3f}, Macro F1-Score: {fscore:.3f}")
        print(
            f"Micro Precision: {micro_precision:.3f}, Micro Recall: {micro_recall:.3f}, Micro F1-Score: {micro_fscore:.3f}")
        print(
            f"Weighted Precision: {weighted_precision:.3f}, Weighted Recall: {weighted_recall:.3f}, Weighted F1-Score: {weighted_fscore:.3f}")
        ##########################
        # print(f"predictions and targets at {timestamp}:{all_preds},\n{all_targets}")
        # print(f"class_probabilities_at {timestamp}:{class_probabilities}")
        # print(f"class_labels_{timestamp}: {class_labels}")
        # print(f"class_accuracies_output{timestamp}: {class_accuracies_output}")
        # print(f"predicted_labels_{timestamp}: {predicted_labels}")
        # print(f"misclassifications_{timestamp}: {misclassifications}")
        # print(f"shot_capacity_changes_{timestamp}: {shot_capacity_changes}")

        data_to_save = {
            "predictions": all_preds,
            "targets": all_targets,
            "class_labels": class_labels,
            "class_probabilities": class_probabilities,
            "class_accuracies": class_accuracies_output,
            "misclassifications": misclassifications,
            "shot_capacity_changes": shot_capacity_changes
        }
        # 尝试保存到 JSON 文件
        try:
            with open(f'{output_dir}/test_data{timestamp}.json', 'w') as f:
                json.dump(data_to_save, f, indent=4)
            print("Data successfully saved to JSON.")
        except TypeError as e:
            print(f"An error occurred: {e}")

        # with open(f'{output_dir}/predictions and targets{timestamp}.json', 'w') as f:
        #     json.dump(data_to_save, f, indent=1)
        # with open(f'{output_dir}/class_probabilities_{timestamp}.json', 'w') as f:
        #     json.dump(class_probabilities, f, indent=4)
        # with open(f'{output_dir}/class_labels_{timestamp}.json', 'w') as f:
        #     json.dump(class_labels, f, indent=4)
        # with open(f'{output_dir}/class_accuracies_for_output{timestamp}.json', 'w') as f:
        #     json.dump(class_accuracies_output, f, indent=4)
        # with open(f'{output_dir}/predicted_labels_{timestamp}.json', 'w') as f:
        #     json.dump(predicted_labels, f, indent=4)
        # with open(f'{output_dir}/misclassifications_{timestamp}.json', 'w') as f:
        #     json.dump(misclassifications, f, indent=4)
        # with open(f'{output_dir}/shot_capacity_changes_{timestamp}.json', 'w') as f:
        #     json.dump(shot_capacity_changes, f, indent=4)

        return sum(accuracies) / len(accuracies), class_accuracies  # 这里新加了一个class的输出


###########################################改进一些代码########################################################
def adjust_shot_capacity(class_accuracies, shot_capacity_dict, lower_threshold=0.3, upper_threshold=0.8,
                         max_capacity=15, min_capacity=2):
    """
    根据类别的准确率动态调整 Shot Capacity。

    :param class_accuracies: 字典，包含每个类别的准确率。
    :param shot_capacity_dict: 字典，存储每个类别的 Shot Capacity。
    :param lower_threshold: 低于该准确率时增加 Shot Capacity。
    :param upper_threshold: 高于该准确率时减少 Shot Capacity。
    :param max_capacity: 最大 Shot Capacity。
    :param min_capacity: 最小 Shot Capacity。
    """
    changes = {}  # 记录变化的类别和其容量变化
    for label, correct_list in class_accuracies.items():
        if len(correct_list) > 0:  # 确保该类别在当前批次中有涉及到
            accuracy = sum(correct_list) / len(correct_list)
            original_capacity = shot_capacity_dict[label]
            if accuracy < lower_threshold:  # 如果准确率低于下限，增加 shot capacity
                shot_capacity_dict[label] = min(shot_capacity_dict[label] + 1, max_capacity)
            elif accuracy > upper_threshold:  # 如果准确率高于上限，减少 shot capacity
                shot_capacity_dict[label] = max(shot_capacity_dict[label] - 1, min_capacity)

                # 如果容量发生变化，则记录
            if shot_capacity_dict[label] != original_capacity:
                changes[label] = shot_capacity_dict[label]


    # 如果有变化，打印并记录到 WandB
    if changes:
        print(f"Shot capacity adjustments made:")
        for label, new_capacity in changes.items():
            print(f"Class {label}: {shot_capacity_dict[label]}")


        #wandb.log({"shot_capacity_changes": changes})


def main():
    args = get_arguments()
    config_path = args.config

    # Initialize CLIP model
    clip_model, preprocess = clip.load(args.backbone)

    clip_model.eval()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    if args.wandb:
        date = datetime.now().strftime("%b%d_%H-%M-%S")
        group_name = f"{args.backbone}_{args.datasets}_{date}"

    # Run TDA on each dataset
    datasets = args.datasets.split('/')
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset.")

        cfg = get_config_file(config_path, dataset_name)
        print("\nRunning dataset configurations:")
        print(cfg, "\n")

        test_loader, classnames, template = build_test_data_loader(dataset_name, args.data_root, preprocess)
        clip_weights = clip_classifier(classnames, template, clip_model)

        # Initialize the shot capacity dict for positive cache
        pos_shot_capacity_dict = {label: cfg['positive']['shot_capacity'] for label in range(len(classnames))}
        neg_shot_capacity = cfg['negative']['shot_capacity']  # Negative cache capacity remains constant

        if args.wandb:
            run_name = f"{dataset_name}"
            run = wandb.init(project="ETTA-CLIP", config=cfg, group=group_name, name=run_name)

        acc, class_accuracires = run_test_tda(cfg['positive'], cfg['negative'], test_loader, clip_model, clip_weights,
                                              pos_shot_capacity_dict, neg_shot_capacity, output_dir="outputs")

        if args.wandb:
            wandb.log({f"{dataset_name}": acc})
            run.finish()




if __name__ == "__main__":
    main()