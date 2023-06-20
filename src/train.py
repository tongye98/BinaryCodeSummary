import torch 
import yaml
import logging 
from typing import Union, Dict, Generator, List
from tqdm import tqdm 
from pathlib import Path 
import random 
import os 
import numpy as np 
import heapq
import math 
import shutil
import codecs
import pickle
import time 
import sentencepiece as spm 
from functools import partial 
from src.model import build_model, Model
from src.data import load_data, BinaryDataset, BinaryData, make_data_loader, PAD_ID
from src.validate import search, eval_accuracies
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AdamW

logger = logging.getLogger(__name__)

def load_config(path: Union[Path,str]="test.yaml") -> Dict:
    if isinstance(path, str):
        path = Path(path)
    with path.open("r", encoding="utf-8") as yamlfile:
        cfg = yaml.safe_load(yamlfile)
    return cfg

def make_logger(model_dir:Path, mode:str):
    logger = logging.getLogger("")
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    
    fh = logging.FileHandler(model_dir/"{}.log".format(mode))
    fh.setLevel(level=logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    logger.info("Hello! This is Tong Ye's Transformer!")

def set_seed(seed:int) -> None:
    """
    Set the random seed for modules torch, numpy and random.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(seed)

def make_model_dir(model_dir:Path, overwrite:bool=False) -> Path:
    model_dir = model_dir.absolute()
    if model_dir.is_dir():
        if not overwrite: # don't allow overwite the directory
            raise FileExistsError(f"<model_dir> {model_dir} exists and don't allow overwrite.")
        else:
            shutil.rmtree(model_dir)
    model_dir.mkdir()
    return model_dir

class TrainManager(object):
    """
    Manage the whole training loop and validation.
    """
    def __init__(self, model:Model, cfg:dict):
        self.cfg = cfg
        self.mode = cfg["model"]["mode"]
        # assert self.mode in {"assembly_comment", "assembly_cfg_comment", "assembly_cfg_pseudo_comment",
        # "assembly_pseudo_comment", "pseudo_comment"}
        # File system 
        train_cfg = cfg["training"]
        self.model_dir = Path(train_cfg["model_dir"])
        assert self.model_dir.is_dir(), "model_dir:{} not found!".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=(self.model_dir/"tensorboard_log").as_posix())

        # logger information
        self.logging_frequency = train_cfg.get("logging_frequence", 100)
        self.valid_frequence = train_cfg.get("validation_frequence", 1000)
        self.store_valid_output = train_cfg.get("store_valid_output", False)
        self.log_valid_samples = train_cfg.get("log_valid_samples", [0, 1, 2])

        # CPU and GPU
        use_cuda = train_cfg["use_cuda"] and torch.cuda.is_available()
        self.n_gpu = torch.cuda.device_count() if use_cuda else 0  
        self.device = torch.device("cuda" if use_cuda else "cpu")
        logger.info("*"*10 + "{} GPUs are used.".format(self.n_gpu) + "*"*10)
        num_workers = train_cfg.get("num_workers", 0)
        if num_workers > 0:
            self.num_workers = min(os.cpu_count(), num_workers)
        logger.info("*"*10 + "{} num_workers are used.".format(self.num_workers) + "*"*10)
        
        # data & batch & epochs
        self.epochs = train_cfg.get("epochs", 100)
        self.shuffle = train_cfg.get("shuffle", True)
        self.max_updates = train_cfg.get("max_updates", np.inf)
        self.batch_size = train_cfg.get("batch_size", 32)
        self.seed = train_cfg.get('random_seed', 980820)

        # model 
        self.model = model
        if self.device.type == "cuda":
            self.model.to(self.device)

        # learning rate and optimization & schedular
        self.learning_rate_min = train_cfg.get("learning_rate_min", 1.0e-8)
        self.clip_grad_function = self.build_gradient_clipper(train_cfg)
        self.optimizer = self.build_optimizer(train_cfg, parameters=self.model.parameters())
        self.scheduler, self.scheduler_step_at = self.build_scheduler(train_cfg, optimizer=self.optimizer)

        # early stop
        self.early_stop_metric = train_cfg.get("early_stop_metric", None)
        if self.early_stop_metric in ["ppl", "loss"]:
            self.minimize_metric = True # lower is better
        elif self.early_stop_metric in ["acc", "bleu"]:
            self.minimize_metric = False # higher is better

        # save / delete checkpoints
        self.num_ckpts_keep = train_cfg.get("num_ckpts_keep", 3)
        self.ckpt_queue = [] # heapq queue List[Tuple[float, Path]]
        
        # initialize training process statistics
        self.train_stats = self.TrainStatistics(
            steps=0, is_min_lr=False, is_max_update=False,
            total_tokens=0, best_ckpt_step=0, minimize_metric=self.minimize_metric,
            best_ckpt_score=np.inf if self.minimize_metric else -np.inf,
        )
        self.train_loader = None

        self.load_model = train_cfg.get("load_model", False)
        if self.load_model:
            self.init_from_checkpoint(Path(self.load_model),
                reset_best_ckpt=train_cfg.get("reset_best_ckpt", False),
                reset_scheduler=train_cfg.get("reset_scheduler", False),
                reset_optimizer=train_cfg.get("reset_optimizer", False),
                reset_iteration_state=train_cfg.get("reset_iteration_state", False))

    def build_gradient_clipper(self, train_cfg:dict):
        """
        Define the function for gradient clipping as specified in configuration.
        Current Options:
            clip_grad_val: clip the gradient if they exceed value.
                torch.nn.utils.clip_grad_value_
            clip_grad_norm: clip the gradients if they norm exceeds this value
                torch.nn.utils.clip_grad_norm_
        """
        clip_grad_function = None 
        if "clip_grad_val" in train_cfg.keys():
            clip_grad_function = partial(torch.nn.utils.clip_grad_value_, clip_value=train_cfg["clip_grad_val"])
        elif "clip_grad_norm" in train_cfg.keys():
            clip_grad_function = partial(torch.nn.utils.clip_grad_norm_, max_norm=train_cfg["clip_grad_norm"])
        return clip_grad_function

    def build_optimizer(self, train_cfg:dict, parameters: Generator) -> torch.optim.Optimizer:
        """
        Create an optimizer for the given parameters as specified in config.
        """
        optimizer_name = train_cfg.get("optimizer", "adam").lower()
        kwargs = {"lr": train_cfg.get("learning_rate", 3.0e-4), "weight_decay":train_cfg.get("weight_decay", 0),
                  "betas": train_cfg.get("adam_betas", (0.9, 0.999)), "eps":train_cfg.get("eps", 1e-8)}
        if optimizer_name == "adam":
            # kwargs: lr, weight_decay; betas
            optimizer = torch.optim.Adam(params=parameters, **kwargs)
        elif optimizer_name == "sgd":
            logger.warning("Use sgd optimizer, please double check.")
            # kwargs: lr, momentum; dampening; weight_decay; nestrov
            optimizer = torch.optim.SGD(parameters, **kwargs)
        elif optimizer_name == 'adamw':
            optimizer_grouped_parameters = [
            {'params': parameters, 'weight_decay': train_cfg.get("weight_decay", 0)}]
            optimizer = AdamW(optimizer_grouped_parameters, lr=train_cfg.get("learning_rate", 3.0e-4), eps=train_cfg.get("eps", 1e-8))
        else:
            assert False, "Invalid optimizer."

        logger.info("%s(%s)", optimizer.__class__.__name__, ", ".join([f"{key}={value}" for key,value in kwargs.items()]))
        return optimizer
    
    def build_scheduler(self, train_cfg:dict, optimizer: torch.optim.Optimizer):
        """
        Create a learning rate scheduler if specified in train config and determine
        when a scheduler step should be executed.
        return:
            - scheduler: scheduler object
            - schedulers_step_at: "validation", "epoch", "step", or "none"
        """
        scheduler, scheduler_step_at = None, None 
        scheduler_name = train_cfg.get("scheduling", None)
        assert scheduler_name in ["StepLR", "ExponentialLR", "ReduceLROnPlateau", "warmup"], "Invalid scheduling."
        
        if scheduler_name == "ReduceLROnPlateau":
            mode = train_cfg.get("mode", "max")
            factor = train_cfg.get("factor", 0.5)
            patience = train_cfg.get("patience", 5)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, 
                        patience=patience, threshold=0.0001, threshold_mode='abs', eps=1e-8)
            scheduler_step_at = "validation"
        elif scheduler_name == "StepLR":
            step_size = train_cfg.get("step_size", 1)
            gamma = train_cfg.get("gamma", 0.1)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            scheduler_step_at = "epoch"  
        elif scheduler_name == "ExponentialLR":
            gamma  = train_cfg.get("gamma", 0.98)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
            scheduler_step_at = "epoch"
        elif scheduler_name == "warmup":
            self.number_train_dataset = 14844
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                    num_warmup_steps=int((self.number_train_dataset / self.batch_size)*self.epochs*0.05), 
                                    num_training_steps=self.epochs*int(self.number_train_dataset / self.batch_size))
            scheduler_step_at = "step"
        else:
            assert False, "Invalid scheduling."
        
        logger.info("Scheduler = %s", scheduler.__class__.__name__)
        return scheduler, scheduler_step_at
        
    def train_and_validate(self, train_dataset:BinaryDataset, valid_dataset:BinaryDataset, vocab_info:Dict):
        """
        Train the model and validate it from time to time on the validation set.
        """
        self.train_loader = make_data_loader(dataset=train_dataset, sampler_seed=self.seed, shuffle=self.shuffle,
                                    batch_size=self.batch_size, num_workers=self.num_workers, mode="train")
        
        # train and validate main loop
        logger.info("Train stats:\n"
                    "\tdevice: %s\n"
                    "\tn_gpu: %d\n"
                    "\tbatch_size: %d",
                    self.device.type, self.n_gpu, self.batch_size)

        try:
            for epoch_no in range(self.epochs):
                logger.info("Epoch %d", epoch_no + 1)
                    
                self.model.train()
                self.model.zero_grad()
                
                # Statistic for each epoch.
                start_time = time.time()
                start_tokens = self.train_stats.total_tokens
                epoch_loss = 0

                for batch_data in self.train_loader:
                    batch_data.to(self.device)
                    batch_length = batch_data.comment_tokens.size(0)
                    sentence_loss, ntokens = self.train_step(batch_data)
                    
                    # reset gradients
                    self.optimizer.zero_grad()

                    sentence_loss.backward()

                    # clip gradients (in-place)
                    if self.clip_grad_function is not None:
                        self.clip_grad_function(parameters=self.model.parameters())
                    
                    # make gradient step
                    self.optimizer.step()

                    # accumulate loss
                    epoch_loss += sentence_loss.item() * batch_length

                    # increment token counter
                    self.train_stats.total_tokens += ntokens
                    
                    # increment step counter
                    self.train_stats.steps += 1
                    if self.train_stats.steps >= self.max_updates:
                        self.train_stats.is_max_update = True
                    
                    # check current leraning rate(lr)
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    # if current_lr < self.learning_rate_min:
                    #     self.train_stats.is_min_lr = True 

                    # log learning process and write tensorboard
                    if self.train_stats.steps % self.logging_frequency == 0:
                        elapse_time = time.time() - start_time
                        elapse_token_num = self.train_stats.total_tokens - start_tokens

                        self.tb_writer.add_scalar(tag="Train/sentence_loss", scalar_value=sentence_loss, global_step=self.train_stats.steps)
                        self.tb_writer.add_scalar(tag="Train/learning_rate", scalar_value=current_lr, global_step=self.train_stats.steps)

                        logger.info("Epoch %3d, Step: %7d, Sentence Loss: %12.6f, Lr: %.6f, Tokens per sec: %6.0f",
                        epoch_no + 1, self.train_stats.steps, sentence_loss, self.optimizer.param_groups[0]["lr"],
                        elapse_token_num / elapse_time)

                        start_time = time.time()
                        start_tokens = self.train_stats.total_tokens
                    
                    # decay learning_rate(lr)
                    if self.scheduler_step_at == "step":
                        self.scheduler.step()

                logger.info("Epoch %3d: total training loss %.2f", epoch_no + 1, epoch_loss)

                if self.scheduler_step_at == "epoch":
                    self.scheduler.step()
                    
                # validate on the entire dev dataset
                if (epoch_no + 1) % self.valid_frequence == 0:
                    valid_duration_time = self.validate(valid_dataset, vocab_info)
                    logger.info("Validation time = {}s.".format(valid_duration_time))

                # check after a number of whole epoch.
                if self.train_stats.is_min_lr or self.train_stats.is_max_update:
                    log_string = (f"minimum learning rate {self.learning_rate_min}" if self.train_stats.is_min_lr else 
                                    f"maximun number of updates(steps) {self.max_updates}")
                    logger.info("Training end since %s was reached!", log_string)
                    break 
            else: # normal ended after training.
                logger.info("Training ended after %3d epoches!", epoch_no + 1)

            logger.info("Best Validation result (greedy) at step %8d: %6.2f %s.",
                        self.train_stats.best_ckpt_step, self.train_stats.best_ckpt_score, self.early_stop_metric)
                    
        except KeyboardInterrupt:
            self.save_model_checkpoint(False, float("nan"))

        self.tb_writer.close()
        return None 

    def train_step(self, batch_data: BinaryData):
        """
        train each batch data.
        """
        assembly_tokens = batch_data.assembly_tokens
        comment_tokens = batch_data.comment_tokens
        pseudo_tokens = batch_data.pseudo_tokens
        pseudo_tokens_codet5 = batch_data.pseudo_tokens_codet5
        cfg_nodes = batch_data.cfg_nodes
        cfg_edges = batch_data.cfg_edges
        cfg_node_batch = batch_data.batch
        # prt = batch_data.ptr
        assembly_tokens_mask = (assembly_tokens != PAD_ID).unsqueeze(1) 
        # assembly_tokens_mask (batch, 1, code_token_length)
        # assembly_tokens_mask: normal is True; pad is False
        pseudo_tokens_mask = (pseudo_tokens != PAD_ID).unsqueeze(1)
        pseudo_tokens_codet5_mask = (pseudo_tokens_codet5 != 0).unsqueeze(1)
        comment_tokens_input = comment_tokens[:, :-1]
        comment_tokens_output = comment_tokens[:, 1:]
        # FIXME why is text_tokens output to make the trget mask
        comment_token_mask = (comment_tokens_output != PAD_ID).unsqueeze(1) 
        # comment_token_mask (batch, 1, trg_length)
        # comment_token_mask: normal is True, pad is False
        ntokens = (comment_tokens_output != PAD_ID).data.sum().item()

        # get loss (run as during training with teacher forcing)
        batch_loss = self.model(return_type="loss_{}".format(self.mode), 
                                assembly_token_input=assembly_tokens,
                                cfg_node_input=cfg_nodes, 
                                cfg_node_batch=cfg_node_batch,
                                edge_index=cfg_edges,
                                trg_input=comment_tokens_input, 
                                trg_truth=comment_tokens_output,
                                assembly_token_mask=assembly_tokens_mask, 
                                cfg_node_mask=None,
                                trg_mask=comment_token_mask,
                                pseudo_token_input=pseudo_tokens,
                                pseudo_token_mask=pseudo_tokens_mask,
                                pseudo_token_codet5_input=pseudo_tokens_codet5,
                                pseudo_token_codet5_mask=pseudo_tokens_codet5_mask) 
        
        batch_length = comment_tokens.size(0)
        sentence_loss = batch_loss / batch_length
        # NOTE sentence_loss is the average-sentence level loss.
        return sentence_loss, ntokens

    def unittest_batch_data(self, code_tokens, ast_nodes, text_tokens, ast_positions, 
            ast_edges, src_mask, text_tokens_input, text_tokens_output, trg_mask, ntokens):
        logger.warning("code token shape = {}".format(code_tokens.size()))
        logger.warning("code token = {}".format(code_tokens))
        logger.warning("ast_nodes shape = {}".format(ast_nodes.size()))
        logger.warning("ast_nodes = {}".format(ast_nodes))
        logger.warning("text token shape = {}".format(text_tokens.size()))
        logger.warning("text token = {}".format(text_tokens))
        logger.warning("ast_position shape = {}".format(ast_positions.size()))
        logger.warning("ast_position = {}".format(ast_positions))
        logger.warning("ast edges shape = {}".format(ast_edges.size()))
        logger.warning("ast_edges = {}".format(ast_edges))
        logger.warning("src_mask shape = {}".format(src_mask.size()))
        logger.warning("src_mask = {}".format(src_mask))
        logger.warning("text_token_input shape = {}".format(text_tokens_input.size()))
        logger.warning("text_token_input = {}".format(text_tokens_input))
        logger.warning("text_tokens_output shape = {}".format(text_tokens_output.size()))
        logger.warning("text_tokens_output = {}".format(text_tokens_output))
        logger.warning("trg_mask shape = {}".format(trg_mask.size()))
        logger.warning("trg_mask = {}".format(trg_mask))
        logger.warning("ntokens = {}".format(ntokens))
        
    def validate(self, valid_dataset: BinaryDataset, vocab_info: Dict):
        """
        Validate on the valid dataset.
        return the validate time.
        """
        validate_start_time = time.time()

        self.valid_loader = make_data_loader(dataset=valid_dataset, sampler_seed=self.seed, shuffle=False,
                    batch_size=self.batch_size, num_workers=self.num_workers, mode="valid")
        
        self.model.eval()

        all_validate_outputs = []
        all_validate_probability = []
        all_validate_attention = []
        all_validate_loss = 0
        validate_score = {"loss":float("nan"), "ppl":float("nan")}
        
        logger.info("Beam Size = {}".format(int(self.cfg["testing"]["beam_size"])))
        for batch_data in tqdm(self.valid_loader, desc="Validating"):
            batch_data.to(self.device)
            with torch.no_grad():
                normalized_batch_loss, ntokens = self.train_step(batch_data)
                all_validate_loss += normalized_batch_loss * self.batch_size

            stacked_output, stacked_probability, stacked_attention = search(batch_data, self.model, self.cfg)

            all_validate_outputs.extend(stacked_output)
            all_validate_probability.extend(stacked_probability if stacked_probability is not None else [])
            all_validate_attention.extend(stacked_attention if stacked_attention is not None else [])
        
        validate_score["loss"] = all_validate_loss / len(valid_dataset)
        validate_score["ppl"] = math.exp(0) # FIXME how to compute ppl

        # all valid_dataset process
        text_vocab = vocab_info["comment_token_vocab"]["self"]
        tokenizer_trg = vocab_info["tokenizer_trg"]
        model_generated = text_vocab.arrays_to_sentences(arrays=all_validate_outputs, cut_at_eos=True, skip_pad=True)
        if self.cfg["data"].get("use_tokenizer") == "robertatokenizer":
            logger.info("use robertatokenizer to decode...")
            if tokenizer_trg is not None :
                logger.info("tokenizer trg is None")
                model_generated = [tokenizer_trg.convert_tokens_to_string(output) for output in model_generated]
            else:
                model_generated = [" ".join(output) for output in model_generated]
        elif self.cfg["data"].get("use_tokenizer") == "sentencepiece_binary_model":
            logger.info("use sentencepice to decode...")
            sp = spm.SentencePieceProcessor(model_file=self.cfg["data"].get("sentencepiece_binary_model"))
            model_generated = [sp.DecodePieces(output) for output in model_generated]
        else:
            logger.info("not use tokenizer to decode...")
            model_generated = [" ".join(output) for output in model_generated]

        target_truth = valid_dataset.target_truth

        validate_duration_time = time.time() - validate_start_time

        bleu, rouge_l, meteor = eval_accuracies(model_generated, target_truth)
        validate_score["bleu"] = bleu
        validate_score["rouge_l"] = rouge_l
        validate_score["meteor"] = meteor

        # write eval_metric and corresponding score to tensorboard
        for eval_metric, score in validate_score.items():
                self.tb_writer.add_scalar(tag=f"valid/{eval_metric}",scalar_value=score, global_step=self.train_stats.steps)
        
        # the most important metric
        ckpt_score = validate_score[self.early_stop_metric]

        # set scheduler
        if self.scheduler_step_at == "validation":
            self.scheduler.step(bleu)

        # update new best
        new_best = self.train_stats.is_best(ckpt_score)
        if new_best:
            logger.info("Horray! New best validation score [%s]!", self.early_stop_metric)
            self.train_stats.best_ckpt_score = ckpt_score
            self.train_stats.best_ckpt_step = self.train_stats.steps
        
        metrics_string = "Bleu = {}, Rouge_L = {}, Meteor = {}".format(bleu, rouge_l, meteor)
        logger.info("Evaluation result({}) {}".format("Greedy Search", metrics_string))
 
        # save checkpoints
        is_better = self.train_stats.is_better(ckpt_score, self.ckpt_queue) if len(self.ckpt_queue) > 0 else True
        if is_better or self.num_ckpts_keep < 0:
            self.save_model_checkpoint(new_best, ckpt_score)
        
        # append to validation report
        self.add_validation_report(valid_scores=validate_score, new_best=new_best)
        self.log_examples(model_generated, target_truth)

        # store validation set outputs
        if self.store_valid_output is True:
            validate_output_path = Path(self.model_dir) / f"{self.train_stats.steps}.hyps"
            self.write_validation_output_to_file(validate_output_path, model_generated)

        # store attention plot for selected valid sentences
        # TODO 

        return validate_duration_time
  
    def add_validation_report(self, valid_scores:dict, new_best:bool):
        """
        Append a one-line report to validation logging file.
        """
        current_lr = self.optimizer.param_groups[0]["lr"]
        valid_file = Path(self.model_dir) / "validation.log"
        with valid_file.open("a", encoding="utf-8") as fg:
            score_string = "\t".join([f"Steps: {self.train_stats.steps:7}"] + 
            [f"{eval_metric}: {score:.2f}" if eval_metric in ["bleu", "meteor", "rouge-l"] 
            else f"{eval_metric}: {score:.2f}" for eval_metric, score in valid_scores.items()] +
            [f"LR: {current_lr:.8f}", "*" if new_best else ""])
            fg.write(f"{score_string}\n") 
    
    def log_examples(self, model_generated: List[str], target_truth:List[str]):
        """
        Log the first self.log_valid_senteces from given examples.
        hypotheses: decoded hypotheses (list of strings)
        references: decoded references (list of strings)
        """
        for id in self.log_valid_samples:
            if id >= len(model_generated):
                continue
            logger.info("Example #%d", id)

            # detokenized text:
            # logger.info("\tSource: %s", dataset[id])
            logger.info("\tReference: %s", target_truth[id])
            logger.info("\tHypothesis: %s", model_generated[id])
    
    class TrainStatistics:
        def __init__(self, steps:int=0, is_min_lr:bool=False,
                        is_max_update:bool=False, total_tokens:int=0,
                        best_ckpt_step:int=0, best_ckpt_score: float=np.inf,
                        minimize_metric: bool=True) -> None:
            self.steps = steps 
            self.is_min_lr = is_min_lr
            self.is_max_update = is_max_update
            self.total_tokens = total_tokens
            self.best_ckpt_step = best_ckpt_step
            self.best_ckpt_score = best_ckpt_score
            self.minimize_metric = minimize_metric
        
        def is_best(self, score) -> bool:
            if self.minimize_metric:
                is_best = score < self.best_ckpt_score
            else: 
                is_best = score > self.best_ckpt_score
            return is_best
        
        def is_better(self, score: float, heap_queue: list):
            assert len(heap_queue) > 0
            if self.minimize_metric:
                is_better = score < heapq.nlargest(1, heap_queue)[0][0]
            else:
                is_better = score > heapq.nsmallest(1, heap_queue)[0][0]
            return is_better
    
    def save_model_checkpoint(self, new_best:bool, score:float) -> None:
        """
        Save model's current parameters and the training state to a checkpoint.
        The training state contains the total number of training steps, 
                                    the total number of training tokens, 
                                    the best checkpoint score and iteration so far, 
                                    and optimizer and scheduler states.
        new_best: for update best.ckpt
        score: Validation score which is used as key of heap queue.
        """        
        model_checkpoint_path = Path(self.model_dir) / f"{self.train_stats.steps}.ckpt"
        model_state_dict = self.model.state_dict()
        global_states = {
            "steps": self.train_stats.steps,
            "total_tokens": self.train_stats.total_tokens,
            "best_ckpt_score": self.train_stats.best_ckpt_score,
            "best_ckpt_step": self.train_stats.best_ckpt_step,
            "model_state": model_state_dict,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
        }
        torch.save(global_states, model_checkpoint_path)

        def symlink_update(target:Path, link_name:Path):
            """
            Find the file that the symlink currently points to, sets it to
            the new target, and return the previous target if it exists.
            """
            if link_name.is_symlink():
                current_link = link_name.resolve()
                link_name.unlink()
                link_name.symlink_to(target)
                return current_link
            else:
                link_name.symlink_to(target)
                return None 

        # how to update queue and keep the best number of ckpt.
        symlink_target = Path(f"{self.train_stats.steps}.ckpt")
        last_path = Path(self.model_dir) / "lastest.ckpt"
        prev_path = symlink_update(symlink_target, last_path)
        best_path = Path(self.model_dir) / "best.ckpt"
        if new_best:
            prev_best = symlink_update(symlink_target, best_path)
            assert best_path.resolve().stem == str(self.train_stats.best_ckpt_step)
        
        def delete_ckpt(path:Path) -> None:
            try:
                logger.info("Delete %s", path)
                path.unlink()  # delete file
            except FileNotFoundError as error:
                logger.warning("Want to delete old checkpoint %s"
                               "but file does not exist. (%s)", path, error)
        
        # push and pop from the heap queue.
        to_delete = None 
        if not math.isnan(score) and self.num_ckpts_keep > 0:
            if len(self.ckpt_queue) < self.num_ckpts_keep: # don't need pop, only push
                heapq.heappush(self.ckpt_queue, (score, model_checkpoint_path))
            else: # ckpt_queue already full.
                if self.minimize_metric: # the smaller score, the better
                    heapq._heapify_max(self.ckpt_queue) # change a list to a max-heap
                    to_delete = heapq._heappop_max(self.ckpt_queue)
                    heapq.heappush(self.ckpt_queue, (score, model_checkpoint_path))
                else: # the higher score, the better
                    to_delete = heapq.heappushpop(self.ckpt_queue, (score, model_checkpoint_path))
            
            if to_delete is not None:
                assert to_delete[1] != model_checkpoint_path  # don't delete the last ckpt, double check
                if to_delete[1].stem != best_path.resolve().stem:
                    delete_ckpt(to_delete[1]) # don't delete the best ckpt
            
            assert len(self.ckpt_queue) <= self.num_ckpts_keep
            if prev_path is not None and prev_path.stem not in [c[1].stem for c in self.ckpt_queue]:
                delete_ckpt(prev_path)

    def init_from_checkpoint(self, path:Path, 
                             reset_best_ckpt:bool=False, reset_scheduler:bool=False,
                             reset_optimizer:bool=False, reset_iteration_state:bool=False):
        """
        Initialize the training from a given checkpoint file.
        The checkpoint file contain not only model parameters, but also 
        scheduler and optimizer states.
        """
        assert path.is_file(), f"model checkpoint {path} not found!"
        model_checkpoint = torch.load(path, map_location=self.device)
        logger.info("Load model from %s done!", path)
        # restore model parameters
        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
        else:
            logger.warning("Reset Optimizer.")
        
        if not reset_scheduler:
            if model_checkpoint["scheduler_state"] is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
        else:
            logger.warning("Reset Scheduler.")

        if not reset_best_ckpt:
            self.train_stats.best_ckpt_score = model_checkpoint["best_ckpt_score"]
            self.train_stats.best_ckpt_step = model_checkpoint["best_ckpt_step"]
        else:
            logger.warning("Reset tracking of the best checkpoints.")
        
        if not reset_iteration_state:
            self.train_stats.steps = model_checkpoint["steps"]
            self.train_stats.total_tokens = model_checkpoint["total_tokens"]
        else:
            logger.info("Reset data_loader (random seed: {%d}).", self.seed)
        
        if self.device.type == "cuda":
            self.model.to(self.device)

    def write_validation_output_to_file(self, path:Path, array: List[str]) -> None:
        """
        Write list of strings to file.
        array: list of strings.
        """
        with path.open("w", encoding="utf-8") as fg:
            for entry in array:
                fg.write(f"{entry}\n")

def train(cfg_file: str):
    """
    Main Training function.
    """
    cfg = load_config(Path(cfg_file))

    model_dir = Path(cfg["training"]["model_dir"])
    overwrite = cfg["training"].get("overwrite", False)
    load_model = cfg["training"].get("load_model", False)
    if load_model is False:
        model_dir = make_model_dir(model_dir, overwrite) # model_dir: absolute path
        make_logger(model_dir, mode="train_validate")
    else:
        make_logger(model_dir, mode="continue_train")

    set_seed(seed=int(cfg["training"].get("random_seed", 980820)))

    # load data
    train_dataset, valid_dataset, test_dataset, vocab_info = load_data(data_cfg=cfg["data"])
    
    # build model
    model = build_model(model_cfg=cfg["model"], vocab_info=vocab_info)

    # training management
    trainer = TrainManager(model=model, cfg=cfg)

    # train model
    trainer.train_and_validate(train_dataset=train_dataset, valid_dataset=valid_dataset, vocab_info=vocab_info)

