#include "internal.h"

static void collect_one_leaf(MctsTree *tree, float *out_features, int *written, int max_entries) {
  CmctsState state = tree->state;
  Node *node = tree->root;
  Node *path[CMCTS_ACTION_SIZE + 1];
  int path_len = 0;
  path[path_len++] = node;

  while (node->expanded && node_child_count(node) > 0 && !state.terminal) {
    int action = -1;
    node = node_select_child(node, tree->c_puct, &action);
    if (!node || action < 0) return;
    state_apply_action(&state, action);
    path[path_len++] = node;
  }

  if (state.terminal) {
    backup(path, path_len, state_outcome_for_player(&state, state.to_play));
    return;
  }
  if (*written >= max_entries) return;
  if (!tree_push_pending_leaf(tree, &state, node, path, path_len)) return;
  state_write_features(&state, out_features + (size_t)(*written) * CMCTS_FEATURE_STRIDE);
  *written += 1;
}

int mcts_batch_collect_leaves(MctsTree *const *trees,
                              int num_trees,
                              int leaves_per_tree,
                              float *out_features,
                              int max_entries) {
  int written = 0;
  if (leaves_per_tree < 1) leaves_per_tree = 1;
  for (int i = 0; i < num_trees; i++) {
    tree_clear_pending_leaves(trees[i]);
  }
  for (int i = 0; i < num_trees; i++) {
    MctsTree *tree = trees[i];
    if (!tree || tree->state.terminal) continue;
    for (int leaf = 0; leaf < leaves_per_tree; leaf++) {
      collect_one_leaf(tree, out_features, &written, max_entries);
    }
  }
  return written;
}

void mcts_batch_feed_leaves(MctsTree *const *trees,
                            int num_trees,
                            const float *priors,
                            const float *values) {
  int offset = 0;
  for (int i = 0; i < num_trees; i++) {
    MctsTree *tree = trees[i];
    if (!tree) continue;
    for (int j = 0; j < tree->pending_leaf_count; j++) {
      PendingLeaf *pending = &tree->pending_leaves[j];
      node_expand(pending->node, &pending->state, priors + (size_t)offset * CMCTS_ACTION_SIZE);
      backup(pending->path, pending->path_len, values[offset]);
      offset += 1;
    }
    tree_clear_pending_leaves(tree);
  }
}
