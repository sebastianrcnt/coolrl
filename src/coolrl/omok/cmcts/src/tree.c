#include "internal.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

Node *node_new(int to_play, float prior) {
  Node *node = (Node *)calloc(1, sizeof(Node));
  if (!node) return NULL;
  node->to_play = to_play;
  node->prior = prior;
  return node;
}

void node_free(Node *node) {
  if (!node) return;
  for (int i = 0; i < CMCTS_ACTION_SIZE; i++) {
    node_free(node->children[i]);
  }
  free(node);
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

void node_expand(Node *node, const CmctsState *state, const float *priors) {
  float total = 0.0f;
  int legal_count = 0;
  for (int i = 0; i < CMCTS_ACTION_SIZE; i++) {
    if (state->board[i] == 0 && !state->terminal) {
      float prior = priors[i] > 0.0f ? priors[i] : 0.0f;
      total += prior;
      legal_count += 1;
    }
  }
  for (int i = 0; i < CMCTS_ACTION_SIZE; i++) {
    node_free(node->children[i]);
    node->children[i] = NULL;
  }
  if (legal_count == 0) {
    node->expanded = 1;
    return;
  }
  for (int i = 0; i < CMCTS_ACTION_SIZE; i++) {
    if (state->board[i] != 0) continue;
    float prior = total <= 0.0f ? 1.0f / (float)legal_count : priors[i] / total;
    node->children[i] = node_new(-state->to_play, prior);
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

static int ensure_pending_capacity(PendingLeaf **items, int *capacity, int required) {
  if (*capacity >= required) return 1;
  int next = *capacity > 0 ? *capacity : 16;
  while (next < required) next *= 2;
  PendingLeaf *resized = (PendingLeaf *)realloc(*items, (size_t)next * sizeof(PendingLeaf));
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
  PendingLeaf *pending = &tree->pending_roots[tree->pending_root_count++];
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
  PendingLeaf *pending = &tree->pending_leaves[tree->pending_leaf_count++];
  pending->state = *state;
  pending->node = node;
  memcpy(pending->path, path, (size_t)path_len * sizeof(Node *));
  pending->path_len = path_len;
  return 1;
}
