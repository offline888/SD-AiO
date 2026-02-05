from collections import OrderedDict

import torch
from torch.nn import functional as F
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger
from basicsr.utils.registry import MODEL_REGISTRY

from .base_model import BaseModel

@MODEL_REGISTRY.register()
class DegNet(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        self.net_dc = build_network(opt["network_dc"])
        self.net_dc = self.model_to_device(self.net_dc)

        load_path_dc = self.opt["path"].get("pretrain_network_dc", None)
        if load_path_dc is not None:
            self.net_dc.load_encoder(load_path_dc)
            '''
            param_key = self.opt["path"].get("param_key_dc", "params")
            self.load_network(
                self.net_dc,
                load_path_dc,
                self.opt["path"].get("strict_load_dc", True),
                param_key,
                self.opt.get("remove_norm", False),
            )
            '''

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        # sself.net_dc.train()
        self.net_dc.set_train(False)

        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")

            import copy
            self.net_dc_ema = copy.deepcopy(self.net_dc)
            self.net_dc_ema.eval()
            self.net_dc_ema = self.net_dc_ema.to(self.device)
            '''
            self.net_dc_ema = build_network(self.opt["network_dc"]).to(self.device)
            self.model_ema(0)
            self.net_dc_ema.eval()
            '''

        self.hook_outputs = list() 

        # define losses
        if train_opt.get("classify_opt"):
            self.cri_classify = build_loss(train_opt["classify_opt"]).to(self.device)
        else:
            self.cri_classify = None

        if self.cri_classify is None:
            raise ValueError("Classify loss is None.")

        # set up optimizers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        for k, v in self.net_dc.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")
        
        optim_type = train_opt["optim_dc"].pop("type")
        self.optimizer_dc = self.get_optimizer(
            optim_type, optim_params, **train_opt["optim_dc"]
        )
        self.optimizers.append(self.optimizer_dc)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device, non_blocking=True)
        if "label" in data:
            self.label = data["label"].to(self.device, non_blocking=True).float()
        
    def optimize_parameters(self, current_iter):
        self.optimizer_dc.zero_grad()

        _, _, _, self.logits = self.net_dc(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        
        if self.cri_classify:
            l_classify = self.cri_classify(self.logits, self.label)
            l_total += l_classify
            loss_dict["l_classify"] = l_classify

        l_total.backward()
        self.optimizer_dc.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def save(self, epoch, current_iter):
        if hasattr(self, "net_dc_ema"):
            self.save_network(
                [self.net_dc, self.net_dc_ema],
                "net_dc",
                current_iter,
                param_key=["params", "params_ema"],
            )
        else:
            self.save_network(self.net_dc, "net_dc", current_iter)
        self.save_training_state(epoch, current_iter)

    def model_ema(self, decay=0.999):
        net_dc = self.get_bare_model(self.net_dc)
        net_dc_params = dict(net_dc.named_parameters())
        net_dc_ema_params = dict(self.net_dc_ema.named_parameters())
        for k in net_dc_ema_params.keys():
            net_dc_ema_params[k].data.mul_(decay).add_(
                net_dc_params[k].data, alpha=1 - decay
            )

    def check_window_size(self, window_size_stats):
        window_size, stats = window_size_stats
        if not (isinstance(window_size, tuple) or isinstance(window_size, list) and not stats):
            return [window_size, True]
        return self.check_window_size([max(window_size), False])

    def pre_test(self):
        net_dc_opt = self.opt.get("network_dc", {})
        if "window_size" not in net_dc_opt:
            return

        _, _, h, w = self.lq.size()
        window_size, _ = self.check_window_size(
            [net_dc_opt.get("window_size", h), False]
        )
        self.scale = self.opt.get("scale", 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.lq = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), "reflect")

    @torch.no_grad()
    def test(self):
        self.net_dc.set_train()
        _, _, _, self.output = self.net_dc(self.lq)
        self.net_dc.eval()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img=False, clamp=True):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img, clamp)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img=False, clamp=True):
        self.net_dc.eval()
        with_metrics = True
        use_pbar = self.opt["val"].get("pbar", False)

        if with_metrics:
            if not hasattr(self, "metric_results"):
                self.metric_results = {"acc": 0.0}
            self._initialize_best_metric_results()
            self.metric_results = {metric: 0 for metric in self.metric_results}

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit="image")

        for idx, val_data in enumerate(dataloader):
            self.feed_data(val_data)
            self.pre_test()
            self.test()

            if with_metrics:
                # 针对多标签分类计算 Acc (阈值 0.5)
                logits = self.output
                gt = self.label
                preds = (torch.sigmoid(logits) > 0.5).float()
                # 计算平均准确率
                acc = (preds == gt).float().mean()
                self.metric_results["acc"] += acc.item()

            torch.cuda.empty_cache()

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f"Test {idx}")
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= idx + 1
                self._update_best_metric_result(metric, self.metric_results[metric], current_iter)
            self._log_validation_metric_values(current_iter, tb_logger)

    def _initialize_best_metric_results(self):
        record = dict()
        record["acc"] = dict(val=0.0, iter=-1)
        self.best_metric_results = record

    def _update_best_metric_result(self, metric, val, current_iter):
        if val >= self.best_metric_results[metric]["val"]:
            self.best_metric_results[metric]["val"] = val
            self.best_metric_results[metric]["iter"] = current_iter

    def _log_validation_metric_values(self, current_iter, tb_logger):
        log_str = f"Validation Degradation Classifier.\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f"metrics/{metric}", value, current_iter)