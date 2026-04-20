#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#define INITIAL_NODE_BLOCK_CAPACITY 1024

struct NodeBlock {
  Node *items;
  int capacity;
  int used;
  NodeBlock *next;
};

static NodeBlock *node_block_new(int capacity) {
  NodeBlock *block = (NodeBlock *)calloc(1, sizeof(NodeBlock));
  if (!block) return NULL;
  block->items = (Node *)calloc((size_t)capacity, sizeof(Node));
  if (!block->items) {
    free(block);
    return NULL;
  }
  block->capacity = capacity;
  return block;
}

Node *tree_node_new(MctsTree *tree, int to_play, float prior) {
  if (!tree) return NULL;
  if (tree->next_node_block_capacity <= 0) {
    tree->next_node_block_capacity = INITIAL_NODE_BLOCK_CAPACITY;
  }
  NodeBlock *block = tree->active_node_block;
  if (!block || block->used >= block->capacity) {
    block = node_block_new(tree->next_node_block_capacity);
    if (!block) return NULL;
    block->next = tree->node_blocks;
    tree->node_blocks = block;
    tree->active_node_block = block;
    if (tree->next_node_block_capacity < 65536) {
      tree->next_node_block_capacity *= 2;
    }
  }
  Node *node = &block->items[block->used++];
  memset(node, 0, sizeof(Node));
  if (!node) return NULL;
  node->to_play = to_play;
  node->prior = prior;
  return node;
}

void tree_reset_nodes(MctsTree *tree) {
  if (!tree) return;
  NodeBlock *block = tree->node_blocks;
  while (block) {
    NodeBlock *next = block->next;
    free(block->items);
    free(block);
    block = next;
  }
  tree->node_blocks = NULL;
  tree->active_node_block = NULL;
  tree->next_node_block_capacity = INITIAL_NODE_BLOCK_CAPACITY;
}

void tree_free_nodes(MctsTree *tree) {
  tree_reset_nodes(tree);
}

int node_child_count(const Node *node) {
  int count = 0;
  if (!node) return 0;
  for (int i = 0; i < CMCTS_ACTION_SIZE; i++) {
    if (node->children[i]) count += 1;
  }
  return count;
}

int node_legal_actions(const Node *node, int32_t *out_actions) {
  int count = 0;
  if (!node) return 0;
  for (int i = 0; i < CMCTS_ACTION_SIZE; i++) {
    if (node->children[i]) {
      if (out_actions) out_actions[count] = i;
      count += 1;
    }
  }
  return count;
}

Node *node_select_child(const Node *node, float c_puct, int *out_action) {
  float sqrt_visits = sqrtf((float)(node->visit_count > 1 ? node->visit_count : 1));
  float best_score = -INFINITY;
  Node *best_child = NULL;
  int best_action = -1;
  for (int action = 0; action < CMCTS_ACTION_SIZE; action++) {
    Node *child = node->children[action];
    if (!child) continue;
    float q = child->visit_count == 0 ? 0.0f : -(child->value_sum / (float)child->visit_count);
    float u = c_puct * child->prior * sqrt_visits / (float)(1 + child->visit_count);
    float score = q + u;
    if (score > best_score) {
      best_score = score;
      best_child = child;
      best_action = action;
    }
  }
  if (out_action) *out_action = best_action;
  return best_child;
}

void node_expand(MctsTree *tree, Node *node, const CmctsState *state, const float *priors) {
  if (!tree || !node || node->expanded) return;
  float total = 0.0f;
  int legal_count = 0;
  for (int i = 0; i < CMCTS_ACTION_SIZE; i++) {
    if (state->board[i] == 0 && !state->terminal) {
      float prior = priors[i] > 0.0f ? priors[i] : 0.0f;
      total += prior;
      legal_count += 1;
    }
  }
  if (legal_count == 0) {
    node->expanded = 1;
    return;
  }
  for (int i = 0; i < CMCTS_ACTION_SIZE; i++) {
    if (state->board[i] != 0) continue;
    float prior = total <= 0.0f ? 1.0f / (float)legal_count : priors[i] / total;
    node->children[i] = tree_node_new(tree, -state->to_play, prior);
  }
  node->expanded = 1;
}

void backup(Node **path, int path_len, float value) {
  for (int i = path_len - 1; i >= 0; i--) {
    path[i]->visit_count += 1;
    path[i]->value_sum += value;
    value = -value;
  }
}

void apply_virtual_loss(Node **path, int path_len, float virtual_loss) {
  for (int i = 0; i < path_len; i++) {
    path[i]->visit_count += 1;
    path[i]->value_sum -= virtual_loss;
  }
}

void revert_virtual_loss(Node **path, int path_len, float virtual_loss) {
  for (int i = 0; i < path_len; i++) {
    path[i]->visit_count -= 1;
    path[i]->value_sum += virtual_loss;
  }
}

static int ensure_pending_capacity(PendingEval **items, int *capacity, int required) {
  if (*capacity >= required) return 1;
  int next = *capacity > 0 ? *capacity : 16;
  while (next < required) next *= 2;
  PendingEval *resized = (PendingEval *)realloc(*items, (size_t)next * sizeof(PendingEval));
  if (!resized) return 0;
  *items = resized;
  *capacity = next;
  return 1;
}

void tree_clear_pending_roots(MctsTree *tree) {
  tree->pending_root_count = 0;
}

void tree_clear_pending_leaves(MctsTree *tree) {
  tree->pending_leaf_count = 0;
}

int tree_push_pending_root(MctsTree *tree, const CmctsState *state, Node *node) {
  if (!ensure_pending_capacity(&tree->pending_roots, &tree->pending_root_capacity,
                               tree->pending_root_count + 1)) {
    return 0;
  }
  PendingEval *pending = &tree->pending_roots[tree->pending_root_count++];
  pending->state = *state;
  pending->node = node;
  pending->path[0] = node;
  pending->path_len = 1;
  return 1;
}

int tree_push_pending_leaf(MctsTree *tree, const CmctsState *state, Node *node,
                           Node **path, int path_len) {
  if (!ensure_pending_capacity(&tree->pending_leaves, &tree->pending_leaf_capacity,
                               tree->pending_leaf_count + 1)) {
    return 0;
  }
  PendingEval *pending = &tree->pending_leaves[tree->pending_leaf_count++];
  pending->state = *state;
  pending->node = node;
  memcpy(pending->path, path, (size_t)path_len * sizeof(Node *));
  pending->path_len = path_len;
  return 1;
}
