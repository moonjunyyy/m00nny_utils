import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Union, List, Callable, TypeVar
from re import Pattern
from ._hook_handler import _MetaPEFTHandler
# import torch.distributed as dist

_RegEx = Union[str, Pattern, List[Union[str, Pattern]]]
_Tensor = TypeVar('_Tensor', bound=Tensor)
_Module = TypeVar('_Module', bound=torch.nn.Module)
_Param = TypeVar('_Param', bound=torch.nn.Parameter)

_PreForwardHook = Callable[[_Module, _Tensor], None]
_ForwardHook = Callable[[_Module, _Tensor, _Tensor], None]
_BackwardHook = Callable[[_Module, _Tensor], None]


class _LayerPromptTuning(_MetaPEFTHandler):
    def __init__(
        self,
        module: _Module,
        embed_dim: int = 768,
        num_heads: int = 12,
        embedding_key: str = 'mean',
        global_pool: str = 'token',
        prompt_fn: str = 'prefix',
        pool_size: int = None,
        prompt_length: int = 5,
        prompt_init: str = 'uniform',
        prompt_key_init: str = 'uniform',
        top_k: int = None,
        batchwise_prompt: bool = False,
        same_key_value: bool = False,
    ):
        super().__init__(module)

        # Let's aquire the necessary modules
        self.prompt_fn = prompt_fn
        for _name, _module in self.module.named_modules():
            if re.match(r'.*[Aa]tte*n.*.*', _name):
                self.attn_module = _module
                break

        self.in_proj_weight = None
        self.in_proj_bias = None
        self.q_proj_weight = None
        self.q_proj_bias = None
        self.k_proj_weight = None
        self.k_proj_bias = None
        self.v_proj_weight = None
        self.v_proj_bias = None

        for _name, _param in self.attn_module.named_parameters():
            if re.match(r'.*(qkv|in_proj).weight', _name):
                self.in_proj_weight = _param
                self._get_qkv = self._single_linear_get_qkv
            elif re.match(r'.*(qkv|in_proj).bias', _name):
                self.in_proj_bias = _param
            elif re.match(r'.*q_proj.weight', _name):
                self.q_proj_weight = _param
                self._get_qkv = self._multiple_linear_get_qkv
            elif re.match(r'.*q_proj.bias', _name):
                self.q_proj_bias = _param
            elif re.match(r'.*k_proj.weight', _name):
                self.k_proj_weight = _param
                self._get_qkv = self._multiple_linear_get_qkv
            elif re.match(r'.*k_proj.bias', _name):
                self.k_proj_bias = _param
            elif re.match(r'.*v_proj.weight', _name):
                self.v_proj_weight = _param
                self._get_qkv = self._multiple_linear_get_qkv
            elif re.match(r'.*v_proj.bias', _name):
                self.v_proj_bias = _param
        self._original_forward = self.attn_module.forward

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.embedding_key = embedding_key
        self.pool_size = pool_size
        self.prompt_length = prompt_length
        self.prompt_init = prompt_init
        self.prompt_key_init = 'uniform'
        self.top_k = top_k
        self.batchwise_prompt = False
        self.same_key_value = same_key_value

        self._similarity = None
        self._query_embed = None

        self.prompt_pool = nn.Parameter(
            (torch.rand(
                self.pool_size, self.prompt_length, self.embed_dim
            ) - 0.5) / (self.embed_dim ** 0.5)
        )
        self.prompt_key = nn.Parameter(
            (torch.rand(
                self.pool_size, self.embed_dim
            ) - 0.5) / (self.embed_dim ** 0.5)
        )
        self.enable()

    def enable(self):
        self._enabled = True
        self.attn_module.forward = self._prompt_forward

    def disable(self):
        self._enabled = False
        self.attn_module.forward = self._original_forward

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        self._task_id = value

    @property
    def query_embed(self):
        return self._query_embed

    @query_embed.setter
    def query_embed(self, value):
        if self.embedding_key == 'mean':
            self._query_embed = value.mean(dim=1)
        elif self.embedding_key == 'cls':
            self._query_embed = value[:, 0, :]
        else:
            raise NotImplementedError

    @property
    def similarity(self):
        return self._similarity

    @similarity.setter
    def similarity(self, value):
        self._similarity = value

    def _prompt_forward(self, _input, *args, **kwargs):
        if self.query_embed is not None and self.prompt_fn == 'prompt':
            selected_prompts = self._prompt_selection(_input)
            _input = torch.cat((_input, selected_prompts), dim=1)
            self._query_embed = None
        B, N, C = _input.shape
        q, k, v = self._get_qkv(_input)
        if self.query_embed is not None and self.prompt_fn == 'prefix':
            selected_prompts = self._prompt_selection(_input)
            _Pl = self.prompt_length // 2
            pk = selected_prompts[:, :_Pl, :].reshape(
                B, _Pl, self.num_heads, -1).permute(0, 2, 1, 3)
            pv = selected_prompts[:, _Pl:, :].reshape(
                B, _Pl, self.num_heads, -1).permute(0, 2, 1, 3)
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)
            self._query_embed = None
        x = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=False,
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.attn_module.out_proj(x)
        return x, None

    def _task_specific_selection(self, _input):
        B, N, C = _input.shape
        curr_task = self.task_id
        selected_prompts = self.prompt_pool[curr_task:curr_task+1]
        selected_prompts = selected_prompts.expand(B, -1, -1)
        selected_key = self.prompt_key[curr_task:curr_task+1]
        key_normalized = F.normalize(
            selected_key, p=2, dim=-1, eps=1e-12)
        embed_normalized = F.normalize(
            self.query_embed, p=2, dim=-1, eps=1e-12)
        similarity = torch.einsum(
            'bd,kd->bk', embed_normalized, key_normalized
        )
        self.similarity = torch.sum(similarity, dim=-1) / B
        return selected_prompts

    def _task_similarity_selection(self, _input):
        B, N, C = _input.shape
        curr_task = self.task_id
        base_key = self.prompt_key[:curr_task+1]
        base_knowledge = self.prompt_pool[:curr_task+1]
        key_normalized = F.normalize(
            base_key, p=2, dim=-1, eps=1e-12)
        embed_normalized = F.normalize(
            self.query_embed, p=2, dim=-1, eps=1e-12)
        similarity = torch.einsum(
            'bd,kd->bk', embed_normalized, key_normalized
        )
        selection = torch.topk(similarity, self.top_k, dim=-1).indices
        selected_prompts = base_knowledge[selection].reshape(
            B, -1, C)
        return selected_prompts

    def _prompt_selection(self, _input):
        B, N, C = _input.shape
        if not self.training:
            return self._task_specific_selection(_input)
        else:
            return self._task_similarity_selection(_input)

    def _multiple_linear_get_qkv(self, _input):
        # q = self.q_proj(_input)
        # k = self.k_proj(_input)
        # v = self.v_proj(_input)
        B, N, C = _input.shape
        q = F.linear(_input, self.q_proj_weight, self.q_proj_bias)
        k = F.linear(_input, self.k_proj_weight, self.k_proj_bias)
        v = F.linear(_input, self.v_proj_weight, self.v_proj_bias)
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        v = v.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        if self.same_key_value:
            v = k
        return q, k, v

    def _single_linear_get_qkv(self, _input):
        B, N, C = _input.shape
        qkv = (
            # self.in_proj(_input)
            F.linear(_input, self.in_proj_weight, self.in_proj_bias)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.same_key_value:
            v = k
        return q, k, v


class _LayerRainbowPrompt(_LayerPromptTuning):
    def __init__(
        self,
        module: _Module,
        embed_dim: int = 768,
        num_heads: int = 12,
        embedding_key: str = 'mean',
        global_pool: str = 'token',
        prompt_fn: str = 'prefix',
        pool_size: int = 10,
        prompt_length: int = 5,
        prompt_init: str = 'uniform',
        prompt_key_init: str = 'uniform',
        top_k: int = None,
        same_key_value: bool = False,
        batchwise_prompt: bool = False,
        KI_iter: int = None,
        D1: int = None,
        D2: int = None,
        use_linear: int = None,
        relation_type: int = None,
        self_attn: bool = False,
    ):
        super().__init__(
            module=module,
            embed_dim=embed_dim,
            num_heads=num_heads,
            embedding_key=embedding_key,
            global_pool=global_pool,
            prompt_fn=prompt_fn,
            pool_size=pool_size,
            prompt_length=prompt_length,
            prompt_init=prompt_init,
            prompt_key_init=prompt_key_init,
            top_k=top_k,
            same_key_value=same_key_value,
            batchwise_prompt=batchwise_prompt,
        )

        self.KI_iter = KI_iter
        self.D1 = D1,
        self.D2 = D2,
        self.use_linear = use_linear
        self.relation_type = relation_type
        self.self_attn = self_attn

        self._task_id = True
        self._warm_up = True
        self._similarity = None

        if self.use_linear:
            self.query_matcher = nn.Linear(embed_dim, D2)
            self.key_matcher = nn.Linear(embed_dim, D2)
            self.value_matcher = nn.Linear(embed_dim, D2)
            self.dense = nn.Linear(D2, embed_dim)
        self.fc1 = nn.Linear(embed_dim, D1)
        self.fc2 = nn.Linear(D1, embed_dim)

        self.stored_rainbow_prompts = nn.Parameter(
            torch.zeros(
                self.pool_size,
                self.prompt_length,
                self.embed_dim
            )
        )

    def set_warm_up(self):
        self._warm_up = True

    def unset_warm_up(self):
        self._warm_up = False

    def _prompt_selection(self, _input):
        if not self.training:
            return self._rainbow_prompt_selection(_input)
        elif self._warm_up:
            return self._task_specific_selection(_input)
        else:
            return self._rainbow_prompt_selection(_input)

    def _rainbow_prompt_selection(self, _input):
        B, N, C = _input.shape
        current_task_embed = self.query_embed
        curr_task = self.task_id
        curr_prompt_len = curr_task * self.prompt_length
        if self.training:
            base_key = self.prompt_key[curr_task:curr_task+1]
            curr_base_knowledge = self.prompt_pool[
                curr_prompt_len: curr_prompt_len + self.prompt_length
            ]
            prev_task = max(0, self.task_id)
            prev_prompt_len = prev_task * self.prompt_length
            prev_base_knowledge = self.prompt_pool[
                0: prev_prompt_len + self.prompt_length
            ].detach().clone()
        else:
            base_key = self.prompt_key[:self.task_id+1]
            curr_base_knowledge = self.prompt_pool[
                curr_prompt_len: curr_prompt_len + self.prompt_length
            ]

        key_normalized = F.normalize(base_key, p=2, dim=-1, eps=1e-12)
        embed_normalized = F.normalize(
            current_task_embed, p=2, dim=-1, eps=1e-12
        )
        similarity = torch.einsum(
            'bd, kd->bk', embed_normalized, key_normalized
        )
        self.similarity = torch.sum(similarity, dim=-1) / B

        if self.training:
            attended_prev = self.task_conditioning_step(
                prev_base_knowledge, key_normalized)
            attended_curr = self.task_conditioning_step(
                curr_base_knowledge, key_normalized)
            evolved_prompts = self.prompt_evolution(
                attended_prev, attended_curr,
                self.embed_dim, self.D1,
            )
            rainbow_prompts = evolved_prompts.mean(
                dim=0, keepdim=True)
            with torch.no_grad():
                self.stored_rainbow_prompts[self.task_id].copy_(
                    rainbow_prompts.squeeze(0))
            rainbow_prompts = rainbow_prompts.expand(B, -1, -1)
        else:
            selected_indices = torch.topk(
                similarity, self.top_k, dim=-1
            ).indices
            rainbow_prompts = self.stored_rainbow_prompts[selected_indices].reshape(
                B, self.prompt_length, C
            )
        return rainbow_prompts

    def task_conditioning_step(self, base_knowledge, current_task_embed):
        if self.relation_type == 'attention':
            N, L, D = base_knowledge.size()
            key_expanded = current_task_embed.reshape(1, 1, -1)
            key_expanded = key_expanded.expand(N, L, -1)
            relevance_scores = torch.einsum(
                'nld,nmd->nlm', base_knowledge, key_expanded
            )
            F.softmax(relevance_scores, dim=-1)
            conditioned_base_knowledge = torch.einsum(
                'nlm,nmd->nld', relevance_scores, base_knowledge)
        else:
            relevance_scores = torch.einsum(
                'nld,d->nl', base_knowledge, current_task_embed)
            relevance_scores = F.sigmoid(relevance_scores)
            conditioned_base_knowledge = torch.einsum(
                'nl,nld->nld', relevance_scores, base_knowledge)

        return conditioned_base_knowledge

    def prompt_evolution(
        self,
        attended_prev,
        attended_curr,
        d_model,
        d_ff,
        dropout=0.1
    ):
        def attention_based_transformation(q, k, v, d_model):
            if self.use_linear:
                q = self.query_matcher(q)
                k = self.key_matcher(k)
                v = self.value_matcher(v)

                scaled_attention_logits = torch.matmul(
                    q, k.transpose(1, 2)
                ) / torch.sqrt(torch.tensor(
                    q.shape[-1], dtype=torch.float32
                ).to(q.device))
                attention_weights = F.softmax(scaled_attention_logits, dim=-1)
                output = torch.matmul(attention_weights, v)

                q_transpose = q.transpose(1, 2)
                k_transpose = k.transpose(1, 2)
                transpose_logits = torch.matmul(
                    q_transpose, k_transpose.transpose(1, 2)
                ) / torch.sqrt(torch.tensor(
                    q_transpose.shape[-1], dtype=torch.float32
                ).to(q.device))
                transpose_weights = F.softmax(transpose_logits, dim=-1)
                output = torch.matmul(
                    transpose_weights, output.transpose(1, 2)).transpose(1, 2)
                output = self.dense(output)
            else:
                scaled_attention_logits = torch.matmul(
                    q, k.transpose(1, 2)
                ) / torch.sqrt(torch.tensor(
                    q.shape[-1], dtype=torch.float32
                ).to(q.device))
                attention_weights = F.softmax(scaled_attention_logits, dim=-1)
                output = torch.matmul(attention_weights, v)
            return output

        def task_guided_alignment(l_index, x, d_model, d_ff):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x

        def evolving(l_index, prev, curr, d_model, d_ff, dropout):
            if self.use_linear:
                attn_output = attention_based_transformation(
                    curr, prev, prev, d_model)
                attn_output = F.dropout(attn_output, dropout, training=True)
                out1 = F.layer_norm(prev + attn_output, [d_model])

                ffn_output = task_guided_alignment(
                    l_index, out1, d_model, d_ff)
                ffn_output = F.dropout(ffn_output, dropout, training=True)
                out2 = F.layer_norm(out1 + ffn_output, [d_model])
                return out2
            else:
                attn_output = attention_based_transformation(
                    curr, prev, prev, d_model)
                attn_output = F.dropout(attn_output, dropout, training=True)
                out1 = F.layer_norm(prev + attn_output, [d_model])
                return out1

        task_wise_evolved_results = []
        if not self.self_attn:
            for KI_layer in range(self.task_id+1):
                if KI_layer == self.task_id:
                    attended_p = attended_curr
                    attended_c = attended_curr
                else:
                    attended_p = attended_prev[
                        KI_layer * self.top_k:
                        KI_layer*self.top_k+self.top_k
                    ]
                    attended_c = attended_curr
                evolved_knowledge = evolving(
                    self.task_id, attended_p, attended_c,
                    d_model, d_ff, dropout
                )
                task_wise_evolved_results.append(evolved_knowledge)
            final_representation = torch.cat(task_wise_evolved_results, dim=0)
        else:
            attended_p, attended_c = attended_curr, attended_curr
            for iteration in range(self.KI_iter):
                evolved_knowledge = evolving(
                    self.task_id, attended_p, attended_c,
                    d_model, d_ff, dropout
                )
                attended_p = evolved_knowledge
            final_representation = evolved_knowledge
        return final_representation


class RainbowPromptHandler(_MetaPEFTHandler):
    def __init__(
        self,
        module: nn.Module,
        num_heads: int = 12,
        embed_dim: int = 768,
        embedding_key: str = 'mean',
        prompt_tune_idx: List[int] = [],
        prefix_tune_idx: List[int] = [],
        global_pool: str = 'token',
        prompt_fn: str = 'prefix',
        pool_size: int = None,
        prompt_length: int = 5,
        prompt_init: str = 'uniform',
        prompt_key_init: str = 'uniform',
        top_k: int = None,
        batchwise_prompt: bool = False,
        same_key_value: bool = False,
        use_linear: int = None,
        KI_iter: int = None,
        D1: int = None,
        D2: int = None,
        relation_type: int = None,
        self_attn_idx:  int = None,
    ):
        super().__init__(module)
        self.prompts = nn.ModuleList()
        for _n, _layer in enumerate(module.encoder.layers):
            if _n in prefix_tune_idx:
                _layer_handler = _LayerRainbowPrompt(
                    _layer,
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    embedding_key=embedding_key,
                    global_pool=global_pool,
                    prompt_fn='prefix',
                    pool_size=pool_size,
                    prompt_length=prompt_length,
                    prompt_init=prompt_init,
                    prompt_key_init=prompt_key_init,
                    top_k=top_k,
                    batchwise_prompt=batchwise_prompt,
                    same_key_value=same_key_value,
                    use_linear=use_linear,
                    KI_iter=KI_iter,
                    D1=D1,
                    D2=D2,
                    relation_type=relation_type,
                    self_attn=_n in self_attn_idx,
                )
                self.prompts.append(_layer_handler)

    def enable(self):
        for prompt in self.prompts:
            prompt.enable()

    def disable(self):
        for prompt in self.prompts:
            prompt.disable()

    def set_warm_up(self):
        for prompt in self.prompts:
            prompt.set_warm_up()

    def unset_warm_up(self):
        for prompt in self.prompts:
            prompt.unset_warm_up()

    @property
    def task_id(self):
        return self._task_id

    @task_id.setter
    def task_id(self, value):
        self._task_id = value
        for prompt in self.prompts:
            prompt.task_id = self._task_id

    @ property
    def query_embed(self):
        return self._query_embed

    @ query_embed.setter
    def query_embed(self, value):
        self._query_embed = value
        for prompt in self.prompts:
            prompt.query_embed = self._query_embed

    @ property
    def similarity(self):
        sim = 0
        for prompt in self.prompts:
            sim += (1 - prompt.similarity).mean()
        return sim / len(self.prompts)

    @ similarity.setter
    def similarity(self, value):
        self._similarity = value
        for prompt in self.prompts:
            prompt.similarity = self._similarity

    def forward(self, x):
        self.disable()
        proj = self.module.conv_proj(x)
        proj = proj.flatten(2).transpose(1, 2)
        proj = torch.cat((self.module.class_token.expand(
            proj.shape[0], -1, -1), proj), dim=1)
        embed = self.module.encoder(proj)
        self.query_embed = embed
        self.enable()
        feat = self.module.encoder(proj)
        pred = self.module.heads(feat[:, 0])
        return pred
