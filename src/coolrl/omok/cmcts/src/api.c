#include "internal.h"

#include <stdlib.h>

MctsTree *mcts_tree_new(float c_puct, int exactly_five) {
  MctsTree *tree = (MctsTree *)calloc(1, sizeof(MctsTree));
  if (!tree) return NULL;
  tree->c_puct = c_puct;
  tree->exactly_five = exactly_five;
  tree->root = node_new(1, 0.0f);
  return tree;
}

void mcts_tree_free(MctsTree *tree) {
  if (!tree) return;
  node_free(tree->root);
  free(tree->pending_roots);
  free(tree->pending_leaves);
  free(tree);
}

void mcts_tree_set_initial(MctsTree *tree,
                           const int8_t *board,
                           int to_play,
                           int last_action,
                           int move_count,
                           int winner,
                           int terminal) {
  if (!tree) return;
  state_init(&tree->state, board, to_play, last_action, move_count, winner, terminal, tree->exactly_five);
  node_free(tree->root);
  tree->root = node_new(to_play, 0.0f);
  tree->root_value = 0.0f;
  tree_clear_pending_roots(tree);
  tree_clear_pending_leaves(tree);
}

void mcts_tree_advance(MctsTree *tree, int action) {
  if (!tree || tree->state.terminal) return;
  Node *next = NULL;
  if (action >= 0 && action < CMCTS_ACTION_SIZE) {
    next = tree->root->children[action];
    tree->root->children[action] = NULL;
  }
  node_free(tree->root);
  state_apply_action(&tree->state, action);
  tree->root = next ? next : node_new(tree->state.to_play, 0.0f);
  tree->root_value = tree->root && tree->root->visit_count > 0
                         ? tree->root->value_sum / (float)tree->root->visit_count
                         : 0.0f;
  tree_clear_pending_roots(tree);
  tree_clear_pending_leaves(tree);
}

int mcts_batch_prepare_roots(MctsTree *const *trees,
                             int num_trees,
                             float *out_features,
                             int max_entries) {
  int written = 0;
  for (int i = 0; i < num_trees; i++) {
    tree_clear_pending_roots(trees[i]);
  }
  for (int i = 0; i < num_trees && written < max_entries; i++) {
    MctsTree *tree = trees[i];
    if (!tree || tree->state.terminal || tree->root->expanded) continue;
    if (!tree_push_pending_root(tree, &tree->state, tree->root)) continue;
    state_write_features(&tree->state, out_features + (size_t)written * CMCTS_FEATURE_STRIDE);
    written += 1;
  }
  return written;
}

void mcts_batch_feed_roots(MctsTree *const *trees,
                           int num_trees,
                           const float *priors,
                           const float *values) {
  int offset = 0;
  for (int i = 0; i < num_trees; i++) {
    MctsTree *tree = trees[i];
    if (!tree) continue;
    for (int j = 0; j < tree->pending_root_count; j++) {
      PendingLeaf *pending = &tree->pending_roots[j];
      node_expand(pending->node, &pending->state, priors + (size_t)offset * CMCTS_ACTION_SIZE);
      tree->root_value = values[offset];
      offset += 1;
    }
    tree_clear_pending_roots(tree);
  }
}

void mcts_batch_apply_root_noise(MctsTree *const *trees,
                                 int num_trees,
                                 const float *noise,
                                 const int32_t *offsets,
                                 float epsilon) {
  for (int i = 0; i < num_trees; i++) {
    MctsTree *tree = trees[i];
    if (!tree || tree->state.terminal || !tree->root) continue;
    int offset = offsets[i];
    int local = 0;
    for (int action = 0; action < CMCTS_ACTION_SIZE; action++) {
      Node *child = tree->root->children[action];
      if (!child) continue;
      child->prior = (1.0f - epsilon) * child->prior + epsilon * noise[offset + local];
      local += 1;
    }
  }
}

void mcts_batch_root_num_legal(MctsTree *const *trees, int num_trees, int32_t *out_counts) {
  for (int i = 0; i < num_trees; i++) {
    MctsTree *tree = trees[i];
    out_counts[i] = (!tree || tree->state.terminal) ? 0 : node_child_count(tree->root);
  }
}

void mcts_batch_get_root_values(MctsTree *const *trees, int num_trees, float *out_values) {
  for (int i = 0; i < num_trees; i++) {
    MctsTree *tree = trees[i];
    if (!tree || tree->state.terminal) {
      out_values[i] = 0.0f;
    } else if (tree->root->visit_count > 0) {
      out_values[i] = tree->root->value_sum / (float)tree->root->visit_count;
    } else {
      out_values[i] = tree->root_value;
    }
  }
}

void mcts_batch_extract_visit_counts(MctsTree *const *trees, int num_trees, float *out_counts) {
  for (int i = 0; i < num_trees; i++) {
    float *row = out_counts + (size_t)i * CMCTS_ACTION_SIZE;
    for (int action = 0; action < CMCTS_ACTION_SIZE; action++) row[action] = 0.0f;
    MctsTree *tree = trees[i];
    if (!tree || tree->state.terminal || !tree->root) continue;
    for (int action = 0; action < CMCTS_ACTION_SIZE; action++) {
      Node *child = tree->root->children[action];
      if (child) row[action] = (float)child->visit_count;
    }
  }
}
