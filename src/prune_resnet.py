import os
from collections import Counter
from datetime import datetime
from itertools import groupby

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm

import src.data as dataset
import src.flop as flop
import src.flops_counter_mask as fcm
import utils.lrp_general6 as lrp_gen
from src.filterprune import FilterPruner
from src.logging import MetricLogger
from src.prune_layer import prune_conv_layer
from src.resnet_kuangliu import ResNet18_kuangliu_c, ResNet50_kuangliu_c

import torch_pruning as tp

import os
import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix


class PruningFineTuner:
    def __init__(self, args, model):
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        self.logger = MetricLogger(
            log_dir=f"runs/{datetime.now().isoformat()}",
            loggers={"wandb", "stdout"},
            args=args,
        )

        self.args = args
        self.setup_dataloaders()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()  # log_softmax + NLL loss
        self.COUNT_ROW = 0
        self.training_epoch = 0
        self.training_step = 0
        self.best_acc = {}
        self.save_loss = False
        self.ratio_pruned_filters = 1.0

    def set_pruner(
        self,
    ):
        if self.args.method_type == "lrp" or self.args.method_type == "weight":
            lrp_params_def1 = {
                "conv2d_ignorebias": True,
                "eltwise_eps": 1e-6,
                "linear_eps": 1e-6,
                "pooling_eps": 1e-6,
                "use_zbeta": True,
            }

            lrp_layer2method = {
                "nn.ReLU": lrp_gen.relu_wrapper_fct,
                "nn.BatchNorm2d": lrp_gen.relu_wrapper_fct,
                "nn.Conv2d": lrp_gen.conv2d_beta0_wrapper_fct,
                "nn.Linear": lrp_gen.linearlayer_eps_wrapper_fct,
                "nn.AdaptiveAvgPool2d": lrp_gen.adaptiveavgpool2d_wrapper_fct,
                "nn.MaxPool2d": lrp_gen.maxpool2d_wrapper_fct,
                "sum_stacked2": lrp_gen.eltwisesum_stacked2_eps_wrapper_fct,
            }
            self.wrapper_model = {
                "resnet18": ResNet18_kuangliu_c,
                "resnet50": ResNet50_kuangliu_c,
            }[self.args.arch.lower()]()
            self.wrapper_model.copyfromresnet(
                self.model,
                lrp_params=lrp_params_def1,
                lrp_layer2method=lrp_layer2method,
            )
            updated_model = self.wrapper_model
        else:
            updated_model = self.model

        if hasattr(self, "pruner"):
            self.pruner.set_model(updated_model)
        else:
            self.pruner = FilterPruner(updated_model, self.args)

    def setup_dataloaders(self):
        kwargs = {"drop_last": True}
        if self.args.cuda:
            kwargs["num_workers"] = max(1, min(4, os.cpu_count()))
            kwargs["pin_memory"] = True

        # Data Acquisition
        get_dataset = {
            "cifar10": dataset.get_cifar10,
            "catsanddogs": dataset.get_catsanddogs,
            "oxfordflowers102": dataset.get_oxfordflowers102,
            "cifar100": dataset.get_cifar100,
            "stanfordcars": dataset.get_stanfordcars,
            "eurosat": dataset.get_eurosat,
            # 'imagenet': dataset.get_imagenet,
        }[self.args.dataset.lower()]
        train_dataset, test_dataset = get_dataset(
            image_size=32  # Use downsampling of first conv layer instead of upsampling of image
            if "cifar" in self.args.dataset and "resnet" in self.args.arch
            else 224
        )
        self.logger.info(
            f"train_dataset:{len(train_dataset)}, test_dataset:{len(test_dataset)}"
        )

        # Data Loader (Input Pipeline)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            **kwargs,
        )

        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            **kwargs,
        )

        self.train_num = len(self.train_loader)
        self.test_num = len(self.test_loader)

    def total_num_filters(self):
        # count total number of filter in every conv layer
        filters = 0
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                if hasattr(module, "output_mask"):
                    filters += module.output_mask.sum()
                else:
                    filters += module.out_channels
        return filters

    def train(self, optimizer=None, epochs=10):
        if optimizer is None:
            if self.args.optimizer == "rmsprop":
                optimizer = optim.RMSprop(
                    self.model.parameters(),
                    lr=self.args.lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay,
                )
            else:
                optimizer = optim.SGD(
                    self.model.parameters(),
                    lr=self.args.lr,
                    momentum=self.args.momentum,
                    weight_decay=self.args.weight_decay,
                )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 4), gamma=0.2
        )

        for i in range(epochs):
            self.logger.add_scalar("train/epoch", i)
            try:  # during training
                self.logger.add_scalar("train/lr", scheduler.get_last_lr()[0])
                self.train_epoch(optimizer=optimizer)
                acc, loss, *_ = self.test()
                self.logger.add_scalars({"test/acc": acc, "test/loss": loss})

                scheduler.step()

                if self.ratio_pruned_filters not in self.best_acc:
                    self.best_acc[self.ratio_pruned_filters] = 0.0

                if acc > self.best_acc[self.ratio_pruned_filters]:
                    self.best_acc[self.ratio_pruned_filters] = acc
                    save_loc = f"{self.logger.log_dir}/{self.args.arch}_{self.args.dataset}_{self.ratio_pruned_filters}.pth"
                    self.logger.info(f"saving model to {save_loc}")
                    torch.save(self.model.state_dict(), save_loc)

            except Exception as e:  # during fine-tuning
                self.logger.info(f"Exception: {e}")
                self.train_epoch(optimizer=optimizer)
                test_accuracy, test_loss, flop_value, param_value = self.test()
                self.df.loc[self.COUNT_ROW] = pd.Series(
                    {
                        # "ratio_pruned": self.ratio_pruned_filters,
                        "test_acc": test_accuracy,
                        "test_loss": test_loss,
                        "flops": flop_value,
                        "params": param_value,
                    }
                )
                self.COUNT_ROW += 1

        self.logger.info("Finished fine tuning")

    def train_epoch(self, optimizer=None, rank_filters=False):
        self.train_loss = 0
        for batch_idx, (data, target) in tqdm(
            enumerate(self.train_loader),
            desc="pruning" if rank_filters else "train",
            total=len(self.train_loader),
            miniters=50,
        ):
            if (
                self.args.limit_train_batches > 0
                and batch_idx > self.args.limit_train_batches
            ):
                break
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            # data, target = Variable(data), Variable(target)
            self.train_batch(optimizer, batch_idx, data, target, rank_filters)

        if (
            self.save_loss and not rank_filters
        ):  # save train_loss only during fine-tuning
            self.dt.loc[self.training_epoch] = pd.Series(
                {
                    "ratio_pruned": self.ratio_pruned_filters,
                    "train_loss": self.train_loss / len(self.train_loader.dataset),
                }
            )
        self.training_epoch += 1
        self.train_loss

    def train_batch(self, optimizer, batch_idx, batch, label, rank_filters):
        self.model.train()
        self.model.zero_grad()
        if optimizer is not None:
            optimizer.zero_grad()

        if rank_filters:  # for pruning
            batch.requires_grad = True
            if (
                self.args.method_type == "lrp" or self.args.method_type == "weight"
            ):  # lrp_based
                output = self.wrapper_model(batch)

                # self.logger.info("Computing LRP")

                # Map the original targets to [0, num_selected_classes)
                T = torch.zeros_like(output)
                for ii in range(len(label)):
                    T[ii, label[ii]] = 1.0

                # Multiply output with target
                lrp_anchor = output * T / (output * T).sum(dim=1, keepdim=True)
                output.backward(lrp_anchor, retain_graph=True)

                self.pruner.compute_filter_criterion(
                    self.layer_type, criterion=self.args.method_type
                )
                loss = self.criterion(output, label)

            else:
                output = self.pruner.model(batch)

                loss = self.criterion(output, label)
                loss.backward()

                self.pruner.compute_filter_criterion(
                    self.layer_type, criterion=self.args.method_type
                )

            if batch_idx % 50 == 0:
                self.logger.debug(
                    f"Loss = {loss.item():.3f}, output.sum() = {output.sum():.2f}, grad.sum() = {batch.grad.sum()}, (step = {batch_idx})"
                )

        else:  # for normal training and fine-tuning
            output = self.model(batch)
            loss = self.criterion(output, label)
            loss.backward()
            optimizer.step()

        if not rank_filters:
            # self.logger.add_scalar("Loss", loss.item(), step=self.training_step)
            self.training_step += 1
        self.train_loss += loss.item()

    def test(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        ctr = 0

        for batch_idx, (data, target) in enumerate(self.test_loader):
            if (
                self.args.limit_test_batches > 0
                and batch_idx > self.args.limit_test_batches
            ):
                break
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)

            self.model.zero_grad()
            with torch.enable_grad():
                output = self.model(data)

            test_loss += self.criterion(output, target).item()
            # Get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            ctr += len(pred)

        test_loss /= ctr
        test_accuracy = float(correct) / ctr
        self.logger.add_scalars({"test/loss": test_loss, "test/acc": test_accuracy})

        if self.save_loss:
            sample_batch = (
                torch.FloatTensor(1, 3, 32, 32).cuda()
                if self.args.cuda
                else torch.FloatTensor(1, 3, 32, 32)
            )
            self.model = fcm.add_flops_counting_methods(self.model)
            self.model.eval().start_flops_count()
            _ = self.model(sample_batch)
            flop_value = self.model.compute_average_flops_cost()
            # flop_value = flop.flops_to_string_value(flop_value)
            param_value = flop.get_model_parameters_number_value_mask(self.model)
            self.model.eval().stop_flops_count()
            self.model = fcm.remove_flops_counting_methods(self.model)
            self.logger.add_scalars(
                {"test/flops": flop_value, "test/params": param_value}
            )

            return test_accuracy, test_loss, flop_value, param_value

        else:
            return test_accuracy, test_loss, None, None

    def get_candidates_to_prune(self, num_filters_to_prune, layer_type="conv"):
        self.layer_type = layer_type

        self.set_pruner()
        self.train_epoch(rank_filters=True)
        self.pruner.normalize_ranks_per_layer()  # Normalization

        ret = self.pruner.get_prunning_plan(num_filters_to_prune)
        self.pruner.reset()
        return ret

    def copy_mask(self):
        my_model = self.model.modules()
        wrapper_model = self.wrapper_model.modules()

        for j in range(len(list(self.wrapper_model.modules()))):
            wrapper_module = next(wrapper_model)
            if hasattr(wrapper_module, "module") and isinstance(
                wrapper_module.module, nn.Conv2d
            ):
                # print(f"wrapper: {wrapper_module.module}")
                for i in range(len(list(self.model.modules()))):
                    my_module = next(my_model)
                    if isinstance(my_module, nn.Conv2d):
                        if hasattr(my_module, "output_mask"):
                            wrapper_module.output_mask = my_module.output_mask
                        break
                    else:
                        continue
            else:
                continue


    def measure_parameters(self):
        # Get class names
        class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']

        # Создание тестовых входных данных (зависит от входного формата модели)
        batch_size = 32
        input_shape = (3, 224, 224)
        test_inputs = torch.randn(batch_size, *input_size, dtype=torch.float32)

        # Измерение задержки
        avg_latency_cpu, latencies = self.measure_latency(test_inputs, device='cpu')
        avg_latency_gpu, latencies = self.measure_latency(test_inputs, device='cuda')

        print(f"Средняя задержка инференса на cpu: {avg_latency_cpu:.2f} ms")
        print(f"Средняя задержка инференса на gpu: {avg_latency_gpu:.2f} ms")
        # Вывод всех замеров
        # print(f"Все замеры задержки: {latencies}")
       
        model_size = self.get_model_size(self.model)
        print(f"Model size: {model_size} MB")
        
        start_time = time.time()
        preds, labels = self.evaluate_model_detailed(self.model, self.test_loader, class_names)
        end_time = time.time()

        # calculation of measured characteristics

        precision = precision_score(labels, preds, average='macro') * 100
        recall = recall_score(labels, preds, average='macro') * 100
        accuracy = ((torch.tensor(preds) == torch.tensor(labels)).sum().item() / len(labels)) * 100

        num_params = sum(p.numel() for p in self.model.parameters())

        elapsed_time = end_time - start_time

        print(f"Total parameters: {num_params:,}")

        print(f'Accuracy: {accuracy:.2f}%')
        print(f"Precision: {precision:.2f}%")
        print(f"Recall: {recall:.2f}%")

        print(f"Execution time: {elapsed_time:.2f} seconds")
                
        example_inputs = torch.randn(1, 3, 224, 224)
        # Move example_inputs to the same device as the model
        example_inputs = example_inputs.to('cpu')  # Move to GPU
        
        macs, nparams = tp.utils.count_ops_and_params(self.model, example_inputs)

        print(f"MACs: {macs/1e9} G, #Params: {nparams/1e6} M")

        return avg_latency_cpu, avg_latency_gpu, model_size, precision, recall, accuracy, macs

    # 15. Detailed Evaluation
    def evaluate_model_detailed(self, dataloader, class_names):
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to('cuda')
                labels = labels.to('cuda')
                
                outputs = self.model(inputs)
                _, preds = torch.max(outputs, 1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Classification Report
        print("Classification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        return all_preds, all_labels

    def get_model_size(self, path="temp_model.pth"):
        torch.save(self.model.state_dict(), path)
        size = os.path.getsize(path) / 1e6  # in MB
        os.remove(path)
        return size

    def measure_latency(self, inputs, warmup_runs=5, measurement_runs=20, device='cpu'):
        """
        Измеряет среднюю задержку инференса модели PyTorch.

        :param model: PyTorch модель.
        :param inputs: Входные данные для модели (тензор или список/кортеж тензоров).
        :param warmup_runs: Количество прогревочных запусков.
        :param measurement_runs: Количество запусков для измерения.
        :param device: Устройство для выполнения инференса ('cpu' или 'cuda').
        :return: Средняя задержка инференса в миллисекундах.
        """
        self.model.eval()

        if isinstance(inputs, (list, tuple)):
            inputs = [inp.to(device) for inp in inputs]
        else:
            inputs = inputs.to(device)
        
        with torch.no_grad():
            # Разогрев (warm-up)
            for _ in range(warmup_runs):
                outputs = self.model(inputs)
                if device == 'cuda':
                    torch.cuda.synchronize()
            
            # Измерение задержки
            latencies = []
            for _ in range(measurement_runs):
                start_time = time.time()
                outputs = self.model(inputs)
                if device == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.time()
                latency = (end_time - start_time) * 1000  # Преобразование в миллисекунды
                latencies.append(latency)
        
        average_latency = sum(latencies) / len(latencies)
        return average_latency, latencies

    def prune(self):
        self.save_loss = True
        self.model.eval()

        # Get the accuracy before pruning
        self.temp = 0
        test_accuracy, test_loss, flop_value, param_value = self.test()

        avg_latency_cpu, avg_latency_gpu, model_size, precision, recall, accuracy, macs = self.measure_parameters()

        # Make sure all the layers are trainable
        # for param in self.model.parameters():
        #     param.requires_grad = True
        self.model.train()

        number_of_filters = self.total_num_filters()
        num_filters_to_prune_per_iteration = int(number_of_filters * self.args.pr_step)
        iterations = int(float(number_of_filters) / num_filters_to_prune_per_iteration)
        iterations = int(iterations * self.args.total_pr)

        self.ratio_pruned_filters = 1.0
        results_file = f"{self.logger.log_dir}/scenario1_results_{self.args.dataset}_{self.args.arch}_{self.args.method_type}_trial{self.args.trialnum:02d}.csv"
        results_file_train = f"{self.logger.log_dir}/scenario1_train_{self.args.dataset}_{self.args.arch}_{self.args.method_type}_trial{self.args.trialnum:02d}.csv"
        self.df = pd.DataFrame(
            columns=["ratio_pruned", "test_acc", "test_loss", "flops", "params"]
        )
        self.dt = pd.DataFrame(columns=["ratio_pruned", "train_loss"])
        self.df.loc[self.COUNT_ROW] = pd.Series(
            {
                "ratio_pruned": self.ratio_pruned_filters,
                "test_acc": test_accuracy,
                "test_loss": test_loss,
                "flops": flop_value,
                "params": param_value,
                "avg_latency_cpu": avg_latency_cpu,
                "avg_latency_gpu": avg_latency_gpu,
                "model_size": model_size,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "macs": macs,
            }
        )
        self.COUNT_ROW += 1
        for kk in range(iterations):
            self.logger.info(f"Ranking filters (iteration {kk+1}/{iterations})")
            prune_targets = self.get_candidates_to_prune(
                num_filters_to_prune_per_iteration, layer_type="conv"
            )
            assert len(prune_targets) == num_filters_to_prune_per_iteration

            model = self.model.cpu()
            ctr = self.total_num_filters()

            # Make sure that there are no duplicate filters
            assert len(set(prune_targets)) == len(prune_targets), [
                x for x in Counter(prune_targets).items() if x[1] > 1
            ]

            grouped_targets = [
                (k, [v for _, v in w])
                for k, w in groupby(sorted(prune_targets), lambda x: x[0])
            ]
            self.logger.info(f"Layers to be pruned: {grouped_targets}")
            for layer_index, filter_inds in grouped_targets:
                model = prune_conv_layer(model, layer_index, filter_inds)

                ctr -= len(filter_inds)
                assert ctr == self.total_num_filters()

            self.model = model.cuda() if self.args.cuda else model
            assert (
                self.total_num_filters()
                == number_of_filters - (kk + 1) * num_filters_to_prune_per_iteration
            ), self.total_num_filters()

            self.ratio_pruned_filters = (
                float(self.total_num_filters()) / number_of_filters
            )
            self.logger.debug(f"Pruning ratio: {self.ratio_pruned_filters}")
            (test_accuracy, test_loss, flop_value, param_value) = self.test()

            (avg_latency_cpu, avg_latency_gpu, model_size, precision, recall, accuracy, macs) = self.measure_parameters()

            metrics = {
                "test/ratio_pruned": self.ratio_pruned_filters,
                "test/acc": test_accuracy,
                "test/loss": test_loss,
                "test/flops": flop_value,
                "test/params": param_value,
                "test/avg_latency_cpu": avg_latency_cpu,
                "test/avg_latency_gpu": avg_latency_gpu,
                "test/model_size": model_size,
                "test/precision": precision,
                "test/recall": recall,
                "test/accuracy": accuracy,
                "test/macs": macs,
            }
            self.df.loc[self.COUNT_ROW] = pd.Series(metrics)
            self.logger.add_scalars(metrics)
            self.COUNT_ROW += 1
            self.df.to_csv(results_file)

            if self.args.method_type == "lrp" or self.args.method_type == "weight":
                self.copy_mask()  # copy mask from my model to the wrapper model

            self.logger.info("Fine tuning to recover from prunning iteration.")
            self.train(epochs=self.args.recovery_epochs)
            self.dt.to_csv(results_file_train)

        self.logger.info("Finished. Going to fine tune the model a bit more")
        self.train(epochs=self.args.recovery_epochs)
        self.dt.to_csv(results_file_train)

        metrics = {
            "test/ratio_pruned": self.ratio_pruned_filters,
            "test/test_acc": test_accuracy,
            "test/test_loss": test_loss,
            "test/flops": flop_value,
            "test/params": param_value,
            "test/avg_latency_cpu": avg_latency_cpu,
            "test/avg_latency_gpu": avg_latency_gpu,
            "test/model_size": model_size,
            "test/precision": precision,
            "test/recall": recall,
            "test/accuracy": accuracy,
            "test/macs": macs,
        }
        self.df.loc[self.COUNT_ROW] = pd.Series(metrics)
        self.logger.add_scalars(metrics, step=self.COUNT_ROW)
        self.df.to_csv(results_file)
