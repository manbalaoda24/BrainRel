"""Budget relation messages without deleting the preserved 100-node bypass."""
import math

import torch
from experts import HypergraphRelationExperts


class BudgetedRelationFilter(HypergraphRelationExperts):
    final_budgets = (5, 9, 2)

    def __init__(self, base, hidden_size, sparse=False):
        super().__init__(base, hidden_size)
        self.sparse = sparse
        self.epoch = 0
        self.retrain_trace = None
        self.requires_grad_(False)
        for module in [self.experts, self.router, self.member_scores, self.edge_gates,
                       self.attention, self.project]:
            module.requires_grad_(True)
        self.queries.requires_grad_(True)
        self.gate_logit.requires_grad_(True)

    def budgets(self):
        sizes = tuple(int((self.edge_types == kind).sum()) for kind in range(3))
        if not self.sparse or self.epoch <= 1:
            return sizes
        fraction = min(1., (self.epoch - 1) / 2.)
        return tuple(math.ceil(size + fraction * (target - size))
                     for size, target in zip(sizes, self.final_budgets))

    def forward(self, nodes):
        encoded = self.encode(nodes.float())
        weights = self.router(encoded).softmax(-1)
        expert_outputs, members_all, gates_all, selections = [], [], [], []
        for kind, budget in enumerate(self.budgets()):
            selected = self.edge_types == kind
            incidence = self.incidence[selected]
            member_logits = self.member_scores[kind](encoded).squeeze(-1).unsqueeze(1)
            members = member_logits.expand(-1, int(selected.sum()), -1)
            members = members.masked_fill(~incidence.unsqueeze(0), -torch.inf).softmax(-1)
            edges = members @ encoded
            raw_logits = self.edge_gates[kind](edges)
            logits = raw_logits.squeeze(-1)
            gate = 2 * raw_logits.sigmoid()
            mask = torch.ones_like(logits)
            messages = edges * gate
            if budget < logits.shape[1]:
                order = torch.argsort(logits, dim=1, descending=True, stable=True)
                hard = torch.zeros_like(logits).scatter_(1, order[:, :budget], 1.)
                if self.training and torch.is_grad_enabled():
                    threshold = logits.gather(1, order[:, budget - 1:budget + 1]).mean(1, keepdim=True).detach()
                    temperature = .5 if self.epoch == 2 else .25
                    soft = ((logits - threshold) / temperature).sigmoid()
                    mask = hard + (soft - soft.detach())
                else:
                    mask = hard
                messages = messages * mask.unsqueeze(-1)
            contexts = self.recipients[selected].T @ messages
            expert_outputs.append(self.experts[kind](contexts))
            members_all.append(members)
            gates_all.append(gate.squeeze(-1))
            selections.append(mask.detach())
        experts = torch.stack(expert_outputs, dim=-2)
        filtered = encoded + (weights.unsqueeze(-1) * experts).sum(-2)
        slots, attention = self.attention(self.queries.expand(nodes.shape[0], -1, -1),
                                          filtered, filtered, need_weights=True)
        self.trace = {'members': torch.cat(members_all, 1).detach(),
                      'gates': torch.cat(gates_all, 1).detach(),
                      'node_router': weights.detach(), 'slot_to_node': attention.detach()}
        self.retrain_trace = {'selection': torch.cat(selections, 1), 'budgets': self.budgets()}
        return self.project(slots) * self.gate_logit.sigmoid(), weights, attention

    def relation_trace(self):
        weights = self.trace['node_router'][:, :, self.edge_types]
        mass = torch.einsum('bsn,bne,en->bse', self.trace['slot_to_node'], weights, self.recipients)
        masked = mass * self.retrain_trace['selection'].unsqueeze(1)
        return {'budgets': list(self.budgets()),
                'active_edges': self.retrain_trace['selection'].sum(1).cpu().tolist(),
                'selection': self.retrain_trace['selection'].cpu().tolist(),
                'mean_selected_edge_routing_mass': masked.mean(1).cpu().tolist(),
                'selected_mass_sum': masked.mean(1).sum(1).cpu().tolist(),
                'interpretation': 'Unnormalized unsigned structural routing mass on selected messages; preserved node bypass excluded; not causal importance'}
