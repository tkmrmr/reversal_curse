from typing import Union, List

import torch
import wandb
from wandb.sdk.wandb_run import Run

from src.models.model import Model
from src.models.common import load_hf_model_and_tokenizer

DEFAULT_EVAL_BATCH_SIZE = None


class LlamaModel(Model):
    def __init__(self, model_name_or_path: str, **kwargs) -> None:
        self.model, self.tokenizer = load_hf_model_and_tokenizer(model_name_or_path)
        self.name = model_name_or_path

    def generate(
        self,
        inputs: Union[str, List[str]],
        max_tokens: int,
        remove_padding: bool = True,
        batch_size: int | None = DEFAULT_EVAL_BATCH_SIZE,
        **kwargs,
    ) -> List[str]:
        if isinstance(inputs, str):
            inputs = [inputs]

        generated_texts = []
        batch_size = batch_size or len(inputs)

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            tokenized_input = self.tokenizer(batch_inputs, padding=True, return_tensors="pt").to(self.model.device)
            input_tokens = tokenized_input.input_ids
            attention_mask = tokenized_input.attention_mask

            with torch.no_grad():
                output_tokens = self.model.generate(
                    input_ids=input_tokens, 
                    attention_mask=attention_mask, 
                    max_new_tokens=max_tokens, 
                    **kwargs
                )
            input_length = tokenized_input.input_ids.shape[1]
            generated_tokens = output_tokens[:, input_length:]
            batch_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

            if remove_padding:
                batch_texts = [text.replace("<pad>", "") for text in batch_texts]

            generated_texts.extend(batch_texts)

        return generated_texts

    def _sum_target_logprobs(self, next_token_logprobs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Extracts the log probabilities of the targets from the log probabilities of the model by masking out all but the
        logits corresponding to the targets.

        :param next_token_logprobs (batch_size, seq): The log probabilities of the model for the next token at each position.
        """
        mask = torch.zeros(next_token_logprobs.shape, device=next_token_logprobs.device)
        # left-shift is because probabilities are shifted one to the left
        mask[:, -targets.shape[1]-1:-1] = (targets != self.tokenizer.pad_token_id)

        logprobs_masked = next_token_logprobs * mask

        return logprobs_masked.sum(dim=-1)


    def _cond_log_prob(self, inputs: List[str], targets: List[List[str]], **kwargs) -> List[List[float]]:
        if isinstance(inputs, str):
            inputs = [inputs]
        if isinstance(targets, str):
            targets = [[targets]]

        # flat_targets = [target[0] for target in targets]

        examples = []
        example_inputs = []
        group_sizes = []

        for inp, target_list in zip(inputs, targets):
            group_sizes.append(len(target_list))
            for target_str in target_list:
                examples.append(inp + target_str)
                example_inputs.append(inp)

        examples_tokenized = self.tokenizer(examples, padding=True, return_tensors="pt")
        inputs_tokenized = self.tokenizer(example_inputs, padding=True, return_tensors="pt")
        examples_tokens = examples_tokenized.input_ids.to(self.model.device)
        examples_attention_mask = examples_tokenized.attention_mask.to(self.model.device)
        inputs_tokens = inputs_tokenized.input_ids.to(self.model.device)
        inputs_attention_mask = inputs_tokenized.attention_mask.to(self.model.device)
        # batchにより位置がずれるのを防ぐためattention maskから位置idを作成
        examples_position_ids = examples_attention_mask.long().cumsum(-1) - 1
        examples_position_ids.masked_fill_(examples_attention_mask == 0, 1)

        with torch.no_grad():
            logits = self.model(
                examples_tokens,
                attention_mask=examples_attention_mask,
                position_ids=examples_position_ids,
                labels=examples_tokens,
            ).logits
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            next_token_logprobs = torch.gather(logprobs[:, :-1], dim=-1, index=examples_tokens[:, 1:].unsqueeze(-1)).squeeze(-1)

        # Mask out prompt and padding tokens, leaving only target-token logprobs.
        target_tokens_mask = torch.zeros_like(next_token_logprobs, dtype=torch.int)
        example_lengths = examples_attention_mask.sum(dim=1)
        example_padding_lengths = examples_attention_mask.shape[1] - example_lengths
        for i, (example_length, padding_length) in enumerate(
            zip(example_lengths, example_padding_lengths)
        ):
            prompt_token_ids = inputs_tokens[i][inputs_attention_mask[i].bool()]
            example_token_ids = examples_tokens[i][examples_attention_mask[i].bool()]
            prefix_length = 0
            for prompt_token_id, example_token_id in zip(prompt_token_ids, example_token_ids):
                if prompt_token_id != example_token_id:
                    break
                prefix_length += 1

            target_start = int(padding_length.item()) + prefix_length
            target_end = int(padding_length.item() + example_length.item())
            # left shift by one because predictions will be one to the left
            target_tokens_mask[i, max(target_start - 1, 0) : target_end - 1] = 1
        relevant_logprobs = next_token_logprobs * target_tokens_mask
        flat_scores = relevant_logprobs.sum(dim=-1)

        results = []
        idx = 0
        for size in group_sizes:
            results.append(flat_scores[idx : idx + size].tolist())
            idx += size

        return results

    def cond_log_prob(
            self, 
            inputs: List[str], 
            targets: List[List[str]], 
            batch_size: int | None = DEFAULT_EVAL_BATCH_SIZE, 
            **kwargs
    ) -> List[List[float]]:
        results = []
        batch_size = batch_size or len(inputs)

        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i : i + batch_size]
            batch_targets = targets[i : i + batch_size]
            batch_results = self._cond_log_prob(batch_inputs, batch_targets, **kwargs)
            results.extend(batch_results)
        return results

    def get_wandb_runs(self, wandb_entity: str, wandb_project: str) -> List[Run]:
        api = wandb.Api()
        runs = api.runs(
            f"{wandb_entity}/{wandb_project}",
            {"config.fine_tuned_model": self.name},
        )
        return runs
