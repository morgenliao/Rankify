from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm
from transformers import T5Tokenizer

from rankify.utils.models.rank_llm.data import Request, Result
from rankify.utils.models.rank_llm.rerank.listwise.listwise_rankllm import ListwiseRankLLM
from rankify.utils.models.rank_llm.rerank.listwise.lit5.model import FiD, FiDCrossAttentionScore
from rankify.utils.models.rank_llm.rerank.rankllm import PromptMode
from tqdm import tqdm  # Import tqdm for progress tracking


class RankFiDDistill(ListwiseRankLLM):
    """
    Implements **RankFiDDistill**, a **listwise ranking approach** leveraging 
    **Fusion-in-Decoder (FiD)** for effective **retriever-reader knowledge distillation**.



    RankFiDDistill utilizes **Fusion-in-Decoder (FiD)** for **reranking retrieved passages** 
    using **multi-document cross-attention**. The model is optimized for **ranking efficiency** 
    and **distilling knowledge** from reader-based models.

    References:
        - **Izacard, G. & Grave, E. (2020)**: *Distilling Knowledge from Reader to Retriever for Question Answering*.
          [Paper](https://arxiv.org/abs/2012.04584)

    Attributes:
        model (str): The **name or path** of the pre-trained **RankFiDDistill** model.
        context_size (int): The **maximum number of passages** used for ranking.
        prompt_mode (PromptMode): Defines the **prompt template** for FiD.
        num_few_shot_examples (int): Number of **few-shot examples** for ranking.
        window_size (int): The **window size** for ranking multiple documents at a time.
        step_size (int): The **step size** for sliding window ranking.
        precision (str): Precision mode (`"float32"`, `"bfloat16"`, `"float16"`).
        device (str): The device to use (`"cuda"` or `"cpu"`).
        batched (bool): Whether to **enable batch processing**.

    Example:
        ```python
        from rankify.dataset.dataset import Document, Question, Context
        from rankify.models.reranking import Reranking

        # Define a query and contexts
        question = Question("What are the effects of climate change?")
        contexts = [
            Context(text="Climate change leads to rising sea levels.", id=0),
            Context(text="Artificial intelligence is transforming industries.", id=1),
            Context(text="Global temperatures are increasing due to CO2 emissions.", id=2),
        ]
        document = Document(question=question, contexts=contexts)

        # Initialize RankFiDDistill Reranker
        model = Reranking(method='lit5dist', model_name='LiT5-Distill-base')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```
    """
    def _post_init(self):
        self._to_precision(self._precision)

    def _tokenize(self, s: str):
        return self._tokenizer(s)

    def _to_precision(self, precision: str) -> None:
        """
        We don't support python12 for now, after python 12, the code should be changed into
        """
        if precision == "float32":
            self._llm = self._llm.float()
        elif precision == "bfloat16":
            self._llm = self._llm.bfloat16()
        elif precision == "float16":
            self._llm = self._llm.float16()

    def __init__(
        self,
        model: str,
        context_size: int = 150,
        prompt_mode: PromptMode = PromptMode.LiT5,  # Placeholder for actual mode
        num_few_shot_examples: int = 0,
        window_size: int = 20,
        step_size: int = 10,
        precision: str = "bfloat16",
        device: str = "cuda",
        batched: bool = False,
    ) -> None:
        """
        Initializes RankFiDDistill for reranking.

        Args:
            model (str): Path or name of the **RankFiDDistill** model.
            context_size (int, optional): Number of passages used for ranking.
            prompt_mode (PromptMode, optional): Defines the **FiD prompt mode**.
            num_few_shot_examples (int, optional): Number of **few-shot examples** used for ranking.
            window_size (int, optional): Defines the **window size** for ranking.
            step_size (int, optional): Defines the **step size** for sliding window ranking.
            precision (str, optional): Precision format (`"float32"`, `"bfloat16"`, `"float16"`).
            device (str, optional): The device for computation (`"cuda"` or `"cpu"`).
            batched (bool, optional): Whether to use **batch processing**.
        """
        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            window_size=window_size,
        )
        self._precision = precision
        self._tokenizer = T5Tokenizer.from_pretrained(model)
        self._llm = FiD.from_pretrained(model).to(device).eval()

        self._device = device

        self._window_size = window_size
        self._stride = step_size

        self._batched = batched

        self._answer_maxlength = len(
            " > ".join(map(lambda x: f"[{x}]", range(1, window_size + 1)))
        )

        self._output_token_estimate = None

        self._post_init()

    def _run_llm_by_length_unified( self, batch_prompts: List[List[str]]) -> List[Tuple[str, int]]:
        if len(batch_prompts) == 0:
            return []

        self._llm.eval()

        batch_size = len(batch_prompts)
        n_passages = len(batch_prompts[0])

        # single batch, unsqueeze
        inputs = {
            k: v.reshape(batch_size, -1).to(self._device)
            for k, v in self._tokenizer(
                [prompt for prompts in batch_prompts for prompt in prompts],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_tokens(),
            ).items()
        }

        with torch.no_grad():
            self._llm.reset_n_passages(n_passages=n_passages)
            outputs = self._llm.generate(
                **inputs,
                max_length=self._answer_maxlength,
                do_sample=False,
            )

        decoded_outputs = [
            self._tokenizer.decode(outputs[i], skip_special_tokens=True)
            for i in range(outputs.shape[0])
        ]

        # all token size should be equal
        return [
            (decoded_output, outputs.shape[1]) for decoded_output in decoded_outputs
        ]

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        use_logits: bool = False,
        use_alpha: bool = False,
        **kwargs
    ) -> List[Result]:
        """
        Reranks documents in batch using RankFiDDistill.

        Args:
            requests (List[Request]): List of **query and candidate passages** for ranking.
            rank_start (int, optional): The **starting rank index**.
            rank_end (int, optional): The **ending rank index**.
            shuffle_candidates (bool, optional): Whether to **shuffle candidates** before ranking.
            logging (bool, optional): Enable logging for debugging.

        Returns:
            List[Result]: **Ranked list of documents**.
        """
        top_k_retrieve: int = kwargs.get("top_k_retrieve", 100)

        window_size: int = kwargs.get("window_size", self._window_size)
        window_size = min(window_size, top_k_retrieve)
        step: int = kwargs.get("step_size", self._stride)

        populate_exec_summary: bool = kwargs.get("populate_exec_summary", False)

        batch_size = kwargs.get("batch_size", 1)
        #print("|||||||||||||||||||||||||||||||")
        if self._batched:
            # reranking using vllm
            if len(set([len(req.candidates) for req in requests])) != 1:
                raise ValueError(
                    "Batched requests must have the same number of candidates"
                )

            result = []

            #with tqdm(range(0, len(requests))) as bar:
            for i in range(0, len(requests), batch_size):
                batch = requests[i : min(i + batch_size, len(requests))]
                batch_result = self.sliding_windows_batched(
                    batch,
                    rank_start=max(rank_start, 0),
                    rank_end=min(
                        rank_end, len(requests[0].candidates)
                    ),  # TODO: Fails arbitrary hit sizes
                    window_size=window_size,
                    step=step,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                    populate_exec_summary=populate_exec_summary,
                )
                result.extend(batch_result)
                #bar.update(len(batch))

            return result
        else:
            # Normal operation mode
            results = []
            for request in requests:
                result = self.sliding_windows(
                    request,
                    rank_start=max(rank_start, 0),
                    rank_end=min(rank_end, len(request.candidates)),
                    window_size=window_size,
                    step=step,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                    populate_exec_summary=populate_exec_summary,
                )
                results.append(result)
            return results

    def run_llm_batched(
        self, prompts: List[List[Dict[str, str]]], **kwargs
    ) -> List[Tuple[str, int]]:
        
        if len(prompts) == 0:
            return []

        # unfortunately, we are not allowed to use VLLM on T5. However, we could unify the prompts by passage size
        #   (which is commonly the same) then rerank stuff having same passage sizes

        prompt_infos = [list(map(lambda x: x["text"], prompt)) for prompt in prompts]

        return self._run_llm_by_length_unified(prompt_infos)

    def create_prompt_batched(
        self, results: List[Result], rank_start: int, rank_end: int, batch_size: int
    ) -> List[Tuple[List[Dict[str, str]], int]]:
        return [self.create_prompt(result, rank_start, rank_end) for result in results]

    def run_llm(self, prompts: List[Dict[str, str]], **kwargs) -> Tuple[str, int]:
        """
        Runs RankFiDDistill to generate ranking predictions.

        Args:
            prompts (List[Dict[str, str]]): **List of query-context pairs** formatted for FiD ranking.

        Returns:
            Tuple[str, int]: **Ranked list of passages**.
        """

        return self._run_llm_by_length_unified(
            [list(map(lambda x: x["text"], prompts))]
        )[0]

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int, use_alpha: bool= False
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Create a prompt based on the result and given ranking range.
        """

        # For now, we concat the prompt, because it seems LiT5 is also concatting the stuff
        prompts = [
            {
                "text": self._gen_passage(
                    result.query["text"],
                    i + 1 - rank_start,
                    self.convert_doc_to_prompt_content(
                        result.candidates[i]["doc"], self.max_tokens()
                    ),
                )
            }
            for i in range(rank_start, rank_end)
        ]

        return prompts, sum(self.get_num_tokens(prompt["text"]) for prompt in prompts)

    def get_num_tokens(self, prompt: Union[str, List[Dict[str, str]]]) -> int:
        """
        Abstract method to calculate the number of tokens contained in the given prompt.
        """
        if isinstance(prompt, str):
            return len(self._tokenizer.encode(prompt))
        elif isinstance(prompt, list):
            return sum(len(self._tokenizer.encode(item["text"])) for item in prompt)
        else:
            raise ValueError(
                "Prompt must be a string or a list of dictionaries with a 'text' key."
            )

    def cost_per_1k_token(self, input_token: bool) -> float:
        return 0

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        if current_window_size is None:
            current_window_size = self._window_size
        if (
            self._output_token_estimate is not None
            and self._window_size == current_window_size
        ):
            return self._output_token_estimate
        else:
            output_token_estimate = (
                len(
                    self._tokenizer.encode(
                        " > ".join([f"[{i + 1}]" for i in range(current_window_size)])
                    )
                )
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = output_token_estimate

            return output_token_estimate

    @staticmethod
    def _gen_passage(query: str, index: int, passage: str) -> str:
        """
        Formats passages for the RankFiDDistill prompt.

        Parameters
        ----------
        query : str
            The search query.
        index : int
            The passage index in the ranking.
        passage : str
            The passage text.

        Returns
        -------
        str
            The formatted passage.
        """
        return f"Search Query: {query} Passage: [{index}] {passage} Relevance Ranking: "


class RankFiDScore(ListwiseRankLLM):
    """
    Implements **RankFiDScore** `[18]_`, a **listwise ranking approach** leveraging 
    **Fusion-in-Decoder (FiD)** with **cross-attention scoring** for accurate ranking.

    .. _[18]: https://arxiv.org/abs/2012.04584

    RankFiDScore utilizes **Fusion-in-Decoder (FiD) models** optimized for **zero-shot listwise ranking** 
    by leveraging **cross-attention weights** for precise **passage relevance estimation**.

    References:
        - **Izacard, G. & Grave, E. (2020)**: *Distilling Knowledge from Reader to Retriever for Question Answering*.
          [Paper](https://arxiv.org/abs/2012.04584)

    Attributes:
        model (str): The **name or path** of the pre-trained **RankFiDScore** model.
        context_size (int): The **maximum number of passages** used for ranking.
        prompt_mode (PromptMode): Defines the **prompt template** for FiD.
        num_few_shot_examples (int): Number of **few-shot examples** for ranking.
        window_size (int): The **window size** for ranking multiple documents at a time.
        step_size (int): The **step size** for sliding window ranking.
        precision (str): Precision mode (`"float32"`, `"bfloat16"`, `"float16"`).
        device (str): The device to use (`"cuda"` or `"cpu"`).

    Examples:
        **Basic Usage:**
        ```python
        from rankify.dataset.dataset import Document, Question, Context
        from rankify.models.reranking import Reranking

        # Define a query and contexts
        question = Question("What are the effects of climate change?")
        contexts = [
            Context(text="Climate change leads to rising sea levels.", id=0),
            Context(text="Artificial intelligence is transforming industries.", id=1),
            Context(text="Global temperatures are increasing due to CO2 emissions.", id=2),
        ]
        document = Document(question=question, contexts=contexts)

        # Initialize RankFiDScore Reranker
        model = Reranking(method='lit5score', model_name='LiT5-Score-base')
        model.rank([document])

        # Print reordered contexts
        print("Reordered Contexts:")
        for context in document.reorder_contexts:
            print(context.text)
        ```
    """
    def _post_init(self):
        # set the overwrite forward cross attention
        self._llm.overwrite_forward_crossattention()
        self._to_precision(self._precision)

    def _tokenize(self, s: str):
        return self._tokenizer(s)

    def _to_precision(self, precision: str) -> None:
        """
        We don't support python12 for now, after python 12, the code should be changed into
        """
        if precision == "float32":
            self._llm = self._llm.float()
        elif precision == "bfloat16":
            self._llm = self._llm.bfloat16()
        elif precision == "float16":
            self._llm = self._llm.float16()

    def __init__(
        self,
        model: str,
        context_size: int = 150,
        prompt_mode: PromptMode = PromptMode.LiT5,  # Placeholder for actual mode
        num_few_shot_examples: int = 0,
        window_size: int = 20,
        step_size: int = 10,
        precision: str = "bfloat16",
        device: str = "cuda",
        batched: bool = False,
    ) -> None:
        """
        Initializes RankFiDScore for reranking.

        Args:
            model (str): Path or name of the **RankFiDScore** model.
            context_size (int, optional): Number of passages to use for ranking.
            prompt_mode (PromptMode, optional): Defines the **FiD prompt mode**.
            num_few_shot_examples (int, optional): Number of **few-shot examples** used for ranking.
            window_size (int, optional): Defines the **window size** for ranking.
            step_size (int, optional): Defines the **step size** for sliding window ranking.
            precision (str, optional): Precision format (`"float32"`, `"bfloat16"`, `"float16"`).
            device (str, optional): The device for computation (`"cuda"` or `"cpu"`).
            batched (bool, optional): Whether to use **batch processing**.
        """

        super().__init__(
            model=model,
            context_size=context_size,
            prompt_mode=prompt_mode,
            num_few_shot_examples=num_few_shot_examples,
            window_size=window_size,
        )
        self._precision = precision
        self._tokenizer = T5Tokenizer.from_pretrained(model)
        self._llm = FiDCrossAttentionScore.from_pretrained(model).to(device).eval()

        self._device = device
        self._window_size = window_size
        self._stride = step_size

        self._batched = batched

        self._output_token_estimate = None

        self._post_init()

    def _run_llm_by_length_unified(
        self, batch_prompts: List[List[Tuple[str, str]]]
    ) -> List[Tuple[str, int]]:
        if len(batch_prompts) == 0:
            return []

        # get arbitrary query (they should be the same)
        queries = [prompts[0][0] for prompts in batch_prompts]
        batch_size = len(batch_prompts)
        n_passages = len(batch_prompts[0])

        inputs = {
            k: v.reshape(batch_size, -1).to(self._device)
            for k, v in self._tokenizer(
                [prompt for prompts in batch_prompts for (_, prompt) in prompts],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_tokens(),
            ).items()
        }

        passage_ids = inputs["input_ids"]
        passage_mask = inputs["attention_mask"]

        with torch.no_grad():
            self._llm.reset_score_storage()

            outputs = self._llm.generate(
                **inputs, max_length=20, do_sample=False, n_passages=n_passages
            )

        output_sequence_lengths = []

        for output in outputs:
            output_length = 0
            for j in range(output.shape[0]):
                if output[j] == FiDCrossAttentionScore.ANSWER_EOS_TOKEN:
                    output_length = j
                    break
            else:
                output_length = outputs.shape[1]
            output_sequence_lengths.append(output_length)

        query_mask_reader = self._tokenizer(
            queries,
            max_length=self.max_tokens(),
            padding="longest",
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )["attention_mask"].bool()

        with torch.no_grad():
            crossattention_scores = self._llm.get_crossattention_scores(
                n_passages,
                ids=passage_ids.to(self._device),
                mask=passage_mask.bool().to(self._device),
                mask_query=query_mask_reader.to(self._device),
                output_sequence_lengths=output_sequence_lengths,
            )
            # only supports normswoquery for now
            crossattention_score: torch.Tensor = crossattention_scores["normswoquery"]
            sorted, idxes = torch.sort(crossattention_score, dim=-1, descending=True)
            idxes = idxes.detach().cpu()

        return [
            (
                " > ".join([f"[{x + 1}]" for x in idxes[i].tolist()]),
                output_sequence_lengths[i] + crossattention_score.shape[1],
            )
            for i in range(idxes.shape[0])
        ]

    def rerank_batch(
        self,
        requests: List[Request],
        rank_start: int = 0,
        rank_end: int = 100,
        shuffle_candidates: bool = False,
        logging: bool = False,
        use_logits: bool = False,
        use_alpha: bool = False,
        **kwargs
    ) -> List[Result]:
        """
        Reranks documents in batch using RankFiDScore.

        Args:
            requests (List[Request]): List of requests containing queries and candidate passages.
            rank_start (int, optional): The starting rank index.
            rank_end (int, optional): The ending rank index.
            shuffle_candidates (bool, optional): Whether to shuffle candidate passages before ranking.
            logging (bool, optional): Enable logging for debugging.

        Returns:
            List[Result]: The reranked documents.
        """
        top_k_retrieve: int = kwargs.get("top_k_retrieve", 100)

        window_size: int = kwargs.get("window_size", self._window_size)
        window_size = min(window_size, top_k_retrieve)
        step: int = kwargs.get("step_size", self._stride)

        populate_exec_summary: bool = kwargs.get("populate_exec_summary", False)

        batch_size = kwargs.get("batch_size", 1)

        if self._batched:
            # reranking using vllm
            if len(set([len(req.candidates) for req in requests])) != 1:
                raise ValueError(
                    "Batched requests must have the same number of candidates"
                )

            result = []

            with tqdm(range(0, len(requests))) as bar:
                for i in range(0, len(requests), batch_size):
                    batch = requests[i : min(i + batch_size, len(requests))]
                    batch_result = self.sliding_windows_batched(
                        batch,
                        rank_start=max(rank_start, 0),
                        rank_end=min(
                            rank_end, len(requests[0].candidates)
                        ),  # TODO: Fails arbitrary hit sizes
                        window_size=window_size,
                        step=step,
                        shuffle_candidates=shuffle_candidates,
                        logging=logging,
                        populate_exec_summary=populate_exec_summary,
                    )
                    result.extend(batch_result)
                    bar.update(len(batch))

            return result
        else:
            # Normal operation mode
            results = []
            for request in requests: #tqdm(
                result = self.sliding_windows(
                    request,
                    rank_start=max(rank_start, 0),
                    rank_end=min(rank_end, len(request.candidates)),
                    window_size=window_size,
                    step=step,
                    shuffle_candidates=shuffle_candidates,
                    logging=logging,
                    populate_exec_summary=populate_exec_summary,
                )
                results.append(result)
            return results

    def run_llm_batched(
        self, prompts: List[List[Dict[str, str]]], **kwargs
    ) -> List[Tuple[str, int]]:
        if len(prompts) == 0:
            return []

        # unfortunately, we are not allowed to use VLLM on T5. However, we could unify the prompts by passage size
        #   (which is commonly the same) then rerank stuff having same passage sizes

        processed_prompts = [
            [(x["query"], x["text"]) for x in prmpt] for prmpt in prompts
        ]

        return self._run_llm_by_length_unified(processed_prompts)

    def create_prompt_batched(
        self, results: List[Result], rank_start: int, rank_end: int, batch_size: int
    ) -> List[Tuple[List[Dict[str, str]], int]]:
        return [self.create_prompt(result, rank_start, rank_end) for result in results]

    def run_llm(self, prompts: List[Dict[str, str]], **kwargs) -> Tuple[str, int]:
        """
        Runs RankFiDScore to generate ranking predictions.

        Args:
            prompts (List[Dict[str, str]]): **List of query-context pairs** formatted for FiD ranking.

        Returns:
            Tuple[str, int]: **Ranked list of passages**.
        """
        return self._run_llm_by_length_unified(
            [[(x["query"], x["text"]) for x in prompts]]
        )[0]

    def create_prompt(
        self, result: Result, rank_start: int, rank_end: int, use_alpha: bool= False
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Creates a **prompt** based on the result and the specified **ranking range**.

        Args:
            result (Result): The result object containing **query and candidate passages**.
            rank_start (int): The **starting rank index**.
            rank_end (int): The **ending rank index**.
            use_alpha (bool, optional): Whether to **apply alpha weighting**.

        Returns:
            Tuple[List[Dict[str, str]], int]: A **list of formatted prompts** and their **token count**.
        """
        query = result.query["text"]
        results = []

        sum_token = 0

        for i in range(rank_start, rank_end):
            results.append(
                {
                    "query": f"question: {query}",
                    "text": self._gen_passage(
                        query,
                        self.convert_doc_to_prompt_content(
                            result.candidates[i]["doc"], self.max_tokens()
                        ),
                    ),
                }
            )
            sum_token += len(self._tokenizer.encode(results[-1]["text"]))

        return results, sum_token

    def get_num_tokens(self, prompt: str) -> int:
        """
        Computes the number of tokens in a given **prompt string**.

        Args:
            prompt (str): The input prompt text.

        Returns:
            int: The number of tokens in the **prompt**.
        """
        return len(self._tokenizer.encode(prompt))

    def cost_per_1k_token(self, input_token: bool) -> float:
        """
        Returns the estimated **cost per 1,000 tokens**.

        Args:
            input_token (bool): Whether to compute for **input tokens**.

        Returns:
            float: The cost per **1,000 tokens**.
        """
        return 0.0

    def num_output_tokens(self, current_window_size: Optional[int] = None) -> int:
        """
        Computes the **number of output tokens** for the current **window size**.

        Args:
            current_window_size (Optional[int], optional): The **size of the current ranking window**.

        Returns:
            int: The estimated **output token count**.
        """
        if current_window_size is None:
            current_window_size = self._window_size
        if (
            self._output_token_estimate is not None
            and self._window_size == current_window_size
        ):
            return self._output_token_estimate
        else:
            output_token_estimate = (
                len(
                    self._tokenizer.encode(
                        " > ".join([f"[{i + 1}]" for i in range(current_window_size)])
                    )
                )
                - 1
            )
            if (
                self._output_token_estimate is None
                and self._window_size == current_window_size
            ):
                self._output_token_estimate = output_token_estimate

            return output_token_estimate

    @staticmethod
    def _gen_passage(query: str, passage: str) -> str:
        """
        Formats passages for the RankFiDScore prompt.

        Args:
            query (str): The search query.
            passage (str): The passage text.

        Returns:
            str: The formatted passage in **RankFiDScore format**.
        """
        return f"question: {query} context: {passage}"