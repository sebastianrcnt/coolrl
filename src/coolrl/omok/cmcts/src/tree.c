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
  node->action_size = tree->action_size;
  node->to_play = to_play;
  node->prior = prior;
  /*
   * Leave children NULL until expansion. On 15x15, preallocating one
   * action_size pointer array per unexpanded node is a large hidden cost.
   */
  return node;
}

static Node *tree_clone_subtree(MctsTree *tree, const Node *source) {
  if (!tree || !source) return NULL;
  Node *node = tree_node_new(tree, source->to_play, source->prior);
  if (!node) return NULL;
  node->visit_count = source->visit_count;
  node->value_sum = source->value_sum;
  node->expanded = source->expanded;
  if (!source->children) return node;

  node->children = (Node **)calloc((size_t)tree->action_size, sizeof(Node *));
  if (!node->children) return NULL;
  for (int action = 0; action < source->action_size; action++) {
    if (!source->children[action]) continue;
    node->children[action] = tree_clone_subtree(tree, source->children[action]);
    if (!node->children[action]) return NULL;
  }
  return node;
}

Node *tree_clone_subtree_to_new_arena(MctsTree *tree, const Node *source) {
  if (!tree || !source) return NULL;

  /*
   * Build the chosen subtree in a temporary fresh arena, restore the old arena
   * handle long enough to free it, then install the fresh arena. This keeps tree
   * reuse for the selected line without retaining abandoned branches.
   */
  NodeBlock *old_node_blocks = tree->node_blocks;
  NodeBlock *old_active_node_block = tree->active_node_block;
  int old_next_node_block_capacity = tree->next_node_block_capacity;

  tree->node_blocks = NULL;
  tree->active_node_block = NULL;
  tree->next_node_block_capacity = INITIAL_NODE_BLOCK_CAPACITY;

  Node *cloned = tree_clone_subtree(tree, source);
  NodeBlock *new_node_blocks = tree->node_blocks;
  NodeBlock *new_active_node_block = tree->active_node_block;
  int new_next_node_block_capacity = tree->next_node_block_capacity;

  if (!cloned) {
    tree_reset_nodes(tree);
    tree->node_blocks = old_node_blocks;
    tree->active_node_block = old_active_node_block;
    tree->next_node_block_capacity = old_next_node_block_capacity;
    return NULL;
  }

  tree->node_blocks = old_node_blocks;
  tree->active_node_block = old_active_node_block;
  tree->next_node_block_capacity = old_next_node_block_capacity;
  tree_reset_nodes(tree);

  tree->node_blocks = new_node_blocks;
  tree->active_node_block = new_active_node_block;
  tree->next_node_block_capacity = new_next_node_block_capacity;
  return cloned;
}

void tree_reset_nodes(MctsTree *tree) {
  if (!tree) return;
  NodeBlock *block = tree->node_blocks;
  while (block) {
    NodeBlock *next = block->next;
    for (int i = 0; i < block->used; i++) {
      free(block->items[i].children);
    }
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
  if (!node || !node->children) return 0;
  for (int i = 0; i < node->action_size; i++) {
    if (node->children[i]) count += 1;
  }
  return count;
}

int node_legal_actions(const Node *node, int32_t *out_actions) {
  int count = 0;
  if (!node || !node->children) return 0;
  for (int i = 0; i < node->action_size; i++) {
    if (node->children[i]) {
      if (out_actions) out_actions[count] = i;
      count += 1;
    }
  }
  return count;
}

Node *node_select_child(const Node *node, float c_puct, int *out_action) {
  if (!node || !node->children) return NULL;
  float sqrt_visits = sqrtf((float)(node->visit_count > 1 ? node->visit_count : 1));
  float best_score = -INFINITY;
  Node *best_child = NULL;
  int best_action = -1;
  for (int action = 0; action < node->action_size; action++) {
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
  for (int i = 0; i < state->action_size; i++) {
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
  if (!node->children) {
    /* Allocate dense child slots only for expanded nodes. */
    node->children = (Node **)calloc((size_t)tree->action_size, sizeof(Node *));
    if (!node->children) return;
  }
  for (int i = 0; i < state->action_size; i++) {
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
    path[i]->value_sum += virtual_loss;
  }
}

void revert_virtual_loss(Node **path, int path_len, float virtual_loss) {
  for (int i = 0; i < path_len; i++) {
    path[i]->visit_count -= 1;
    path[i]->value_sum -= virtual_loss;
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
