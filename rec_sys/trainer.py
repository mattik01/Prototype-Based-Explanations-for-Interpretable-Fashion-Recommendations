import os
import time

import torch
import tempfile

from torch import nn
from torch.utils import data

from feature_extraction.feature_extractor_factories import FeatureExtractorFactory
from feature_extraction.feature_ids import inject_feature_ids
from rec_sys.rec_sys import RecSys
from utilities.consts import OPTIMIZING_METRIC, MAX_PATIENCE
from utilities.eval import Evaluator


class Trainer:

    def __init__(self, train_loader: data.DataLoader, val_loader: data.DataLoader, conf, use_ray=True):
        """
        Train and Evaluate the model.
        :param train_loader: Training DataLoader (check music4all_data.Music4AllDataset for more info)
        :param val_loader: Validation DataLoader (check music4all_data.Music4AllDataset for more info)
        :param conf: Experiment configuration parameters
        :param use_ray: If True, use Ray Tune for reporting and checkpointing. If False, use plain torch.save.
        """

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.rec_sys_param = conf.rec_sys_param
        self.ft_ext_param = conf.ft_ext_param
        self.optim_param = conf.optim_param

        self.n_epochs = conf.n_epochs
        self.loss_func_name = conf.loss_func_name
        self.loss_func_aggr = conf.loss_func_aggr if 'loss_func_aggr' in conf else 'mean'

        self.device = conf.device
        self.use_ray = use_ray

        self.optimizing_metric = getattr(conf, '_optimizing_metric', OPTIMIZING_METRIC)
        self.max_patience = getattr(conf, '_max_patience', MAX_PATIENCE)
        # min_delta: minimum improvement on optimizing_metric required to reset patience.
        # 0.0 preserves the original "any improvement counts" behavior when not set explicitly.
        self.min_delta = float(getattr(conf, '_min_delta', 0.0))

        self.model = self._build_model()
        # dc06 fI′ (A11 gauge snap): modules that project their metadata rows onto the μ=0
        # gauge slice after each optimizer step. Cached once — empty for every non-centred
        # model, so the train loop's per-step overhead is an empty-list iteration.
        self._gauge_snap_modules = [m for m in self.model.modules()
                                    if getattr(m, 'center_fields', False) and hasattr(m, 'gauge_snap')]
        self.optimizer = self._build_optimizer()

        print(f'Built Trainer module \n'
              f'- n_epochs: {self.n_epochs} \n'
              f'- loss_func_name: {self.loss_func_name} \n'
              f'- loss_func_aggr: {self.loss_func_aggr} \n'
              f'- device: {self.device} \n'
              f'- optimizing_metric: {self.optimizing_metric} \n'
              f'- max_patience: {self.max_patience} \n'
              f'- min_delta: {self.min_delta} \n')

    def _build_model(self):
        # Step 1 --- Building User and Item Feature Extractors
        n_users = self.train_loader.dataset.n_users
        n_items = self.train_loader.dataset.n_items
        # P6: for feature_item_proto, build the feature_ids tensor from the dataset's item_features.csv
        # (no-op for every other ft_type). Non-mutating: the tensor goes only into the param the factory
        # consumes here, never back into self.ft_ext_param / the serialized config.
        ft_ext_param = inject_feature_ids(self.ft_ext_param, self.train_loader.dataset.data_path)
        user_feature_extractor, item_feature_extractor = \
            FeatureExtractorFactory.create_models(ft_ext_param, n_users, n_items)
        # Step 2 --- Building RecSys Module
        rec_sys = RecSys(n_users, n_items, self.rec_sys_param, user_feature_extractor, item_feature_extractor,
                         self.loss_func_name, self.loss_func_aggr)

        rec_sys.init_parameters()
        rec_sys = nn.DataParallel(rec_sys)
        rec_sys = rec_sys.to(self.device)

        return rec_sys

    def _build_optimizer(self):
        self.lr = self.optim_param['lr'] if 'lr' in self.optim_param else 1e-3
        self.wd = self.optim_param['wd'] if 'wd' in self.optim_param else 1e-4

        optim_name = self.optim_param['optim']
        if optim_name == 'adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        elif optim_name == 'adagrad':
            optim = torch.optim.Adagrad(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        else:
            raise ValueError('Optimizer not yet included')

        print(f'Built Optimizer  \n'
              f'- name: {optim_name} \n'
              f'- lr: {self.lr} \n'
              f'- wd: {self.wd} \n')

        return optim

    def _report(self, metrics, checkpoint_dir=None):
        """Report metrics via Ray Tune (+ W&B if active) or print locally."""
        # Log to W&B if a run is active (initialized in start_training).
        # Never let a W&B failure (missing import, or an online network/comm error
        # mid-training) abort the run — observability must not be fatal.
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics)
        except Exception:
            pass

        if self.use_ray:
            try:
                from ray.tune import report as tune_report
                from ray.tune import Checkpoint as TuneCheckpoint
                if checkpoint_dir is not None:
                    tune_report(metrics, checkpoint=TuneCheckpoint.from_directory(checkpoint_dir))
                else:
                    tune_report(metrics)
            except (ImportError, AttributeError):
                from ray import train
                if checkpoint_dir is not None:
                    train.report(metrics, checkpoint=train.Checkpoint.from_directory(checkpoint_dir))
                else:
                    train.report(metrics)
        else:
            print(f'  Metrics: { {k: (f"{v:.4f}" if isinstance(v, (int, float)) else v) for k, v in metrics.items()} }')

    def run(self, checkpoint_dir=None):
        """
        Runs the Training procedure.
        :param checkpoint_dir: When use_ray=False, directory to save best model checkpoint.
                               Defaults to 'Master/temp/checkpoints/'.
        """
        if not self.use_ray and checkpoint_dir is None:
            checkpoint_dir = os.path.join(os.path.dirname(__file__), '..', 'Master', 'temp', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)

        metrics_values = self.val()
        best_value = metrics_values[self.optimizing_metric]
        self._report({**metrics_values, 'epoch': -1, 'lr': self.lr})
        print('Init - Avg Val Value {:.3f} \n'.format(best_value))

        train_start = time.time()
        epoch_durations = []
        patience = 0
        for epoch in range(self.n_epochs):

            if patience == self.max_patience:
                print('Max Patience reached, stopping.')
                break

            epoch_start = time.time()
            self.model.train()

            epoch_train_loss = 0

            for u_idxs, i_idxs, labels in self.train_loader:
                u_idxs = u_idxs.to(self.device)
                i_idxs = i_idxs.to(self.device)
                labels = labels.to(self.device)

                out = self.model(u_idxs, i_idxs)

                loss = self.model.module.loss_func(out, labels)

                epoch_train_loss += loss.item()

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                for m in self._gauge_snap_modules:
                    m.gauge_snap()

            epoch_train_loss /= len(self.train_loader)

            # Timing metrics
            epoch_duration = time.time() - epoch_start
            epoch_durations.append(epoch_duration)
            elapsed = time.time() - train_start
            avg_epoch_time = elapsed / (epoch + 1)
            # ETA only meaningful when patience is active (model not improving)
            if patience > 0:
                remaining_epochs = min(self.n_epochs - epoch - 1, self.max_patience - patience)
                eta_min = round(avg_epoch_time * remaining_epochs / 60, 1)
            else:
                eta_min = None

            timing_metrics = {
                'epoch': epoch,
                'epoch_duration_s': round(epoch_duration, 1),
                'elapsed_s': round(elapsed, 1),
                'eta_min': eta_min,
                'lr': self.optimizer.param_groups[0]['lr'],
                'patience': patience,
            }

            eta_str = "~{:.1f}min".format(eta_min) if eta_min is not None else "improving"
            print("Epoch {} - Train Loss {:.3f} - {:.0f}s (ETA {})\n".format(
                epoch, epoch_train_loss, epoch_duration, eta_str))

            metrics_values = self.val()
            curr_value = metrics_values[self.optimizing_metric]
            print('Epoch {} - Avg Val Value {:.3f} \n'.format(epoch, curr_value))

            report_metrics = {**metrics_values, 'epoch_train_loss': epoch_train_loss, **timing_metrics}

            if curr_value > best_value + self.min_delta:
                best_value = curr_value
                print('Epoch {} - New best model found (val value {:.3f}, Δ>{:.1e}) \n'.format(
                    epoch, curr_value, self.min_delta))
                if self.use_ray:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        torch.save(self.model.module.state_dict(), os.path.join(tmpdir, 'best_model.pth'))
                        self._report(report_metrics, checkpoint_dir=tmpdir)
                else:
                    torch.save(self.model.module.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
                    self._report(report_metrics)
                patience = 0
            else:
                self._report(report_metrics)
                patience += 1

    @torch.no_grad()
    def val(self):
        """
        Runs the evaluation procedure.
        :return: A scalar float value, output of the validation (e.g. NDCG@10).
        """
        self.model.eval()
        print('Validation started')
        val_loss = 0
        # Expected rows = the dataset's declared eval-row count (== n_users on canonical splits,
        # checked in ProtoRecDataset; fewer on the cold variant) — F-S0-03 divisor semantics.
        eval = Evaluator(self.val_loader.dataset.coo_matrix.nnz)

        for u_idxs, i_idxs, labels in self.val_loader:
            u_idxs = u_idxs.to(self.device)
            i_idxs = i_idxs.to(self.device)
            labels = labels.to(self.device)

            out = self.model(u_idxs, i_idxs)

            val_loss += self.model.module.loss_func(out, labels).item()

            out = nn.Sigmoid()(out)
            out = out.to('cpu')

            eval.eval_batch(out)

        val_loss /= len(self.val_loader)
        metrics_values = {**eval.get_results(), 'val_loss': val_loss}

        return metrics_values
