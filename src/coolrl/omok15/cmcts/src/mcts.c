#include "internal.h"

#ifndef _WIN32
#include <pthread.h>
#endif
#include <stdlib.h>
#include <string.h>

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
    if (!state_apply_action(&state, action)) return;
    path[path_len++] = node;
  }

  if (state.terminal) {
    backup(path, path_len, state_outcome_for_player(&state, state.to_play));
    return;
  }
  if (*written >= max_entries) return;
  apply_virtual_loss(path, path_len, tree->virtual_loss);
  if (!tree_push_pending_leaf(tree, &state, node, path, path_len)) {
    revert_virtual_loss(path, path_len, tree->virtual_loss);
    return;
  }
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

typedef struct {
  MctsTree *const *trees;
  int start;
  int end;
  int leaves_per_tree;
  float *out_features;
  int max_entries;
  int *counts;
} CollectWorkerArgs;

static void collect_tree_range(CollectWorkerArgs *args) {
  for (int i = args->start; i < args->end; i++) {
    MctsTree *tree = args->trees[i];
    if (!tree || tree->state.terminal) continue;

    int segment_start = i * args->leaves_per_tree;
    if (segment_start >= args->max_entries) continue;
    int segment_capacity = args->leaves_per_tree;
    if (segment_start + segment_capacity > args->max_entries) {
      segment_capacity = args->max_entries - segment_start;
    }

    int written = 0;
    float *segment = args->out_features + (size_t)segment_start * CMCTS_FEATURE_STRIDE;
    for (int leaf = 0; leaf < args->leaves_per_tree; leaf++) {
      collect_one_leaf(tree, segment, &written, segment_capacity);
    }
    args->counts[i] = written;
  }
}

#ifndef _WIN32
static void *collect_tree_range_thread(void *payload) {
  collect_tree_range((CollectWorkerArgs *)payload);
  return NULL;
}
#endif

int mcts_batch_collect_leaves_threaded(MctsTree *const *trees,
                                       int num_trees,
                                       int leaves_per_tree,
                                       float *out_features,
                                       int max_entries,
                                       int num_threads) {
#ifdef _WIN32
  (void)num_threads;
  return mcts_batch_collect_leaves(trees, num_trees, leaves_per_tree, out_features, max_entries);
#else
  if (leaves_per_tree < 1) leaves_per_tree = 1;
  if (num_threads <= 1 || num_trees <= 1) {
    return mcts_batch_collect_leaves(trees, num_trees, leaves_per_tree, out_features, max_entries);
  }
  if (num_threads > num_trees) num_threads = num_trees;

  int *counts = (int *)calloc((size_t)num_trees, sizeof(int));
  pthread_t *threads = (pthread_t *)calloc((size_t)num_threads, sizeof(pthread_t));
  CollectWorkerArgs *args = (CollectWorkerArgs *)calloc((size_t)num_threads, sizeof(CollectWorkerArgs));
  if (!counts || !threads || !args) {
    free(counts);
    free(threads);
    free(args);
    return mcts_batch_collect_leaves(trees, num_trees, leaves_per_tree, out_features, max_entries);
  }

  for (int i = 0; i < num_trees; i++) {
    tree_clear_pending_leaves(trees[i]);
  }

  int launched = 0;
  for (int thread_idx = 0; thread_idx < num_threads; thread_idx++) {
    int start = (num_trees * thread_idx) / num_threads;
    int end = (num_trees * (thread_idx + 1)) / num_threads;
    args[thread_idx] = (CollectWorkerArgs){
        .trees = trees,
        .start = start,
        .end = end,
        .leaves_per_tree = leaves_per_tree,
        .out_features = out_features,
        .max_entries = max_entries,
        .counts = counts,
    };
    if (pthread_create(&threads[thread_idx], NULL, collect_tree_range_thread, &args[thread_idx]) != 0) {
      for (int i = 0; i < launched; i++) {
        pthread_join(threads[i], NULL);
      }
      for (int remaining = thread_idx; remaining < num_threads; remaining++) {
        collect_tree_range(&args[remaining]);
      }
      launched = 0;
      break;
    }
    launched += 1;
  }

  for (int i = 0; i < launched; i++) {
    pthread_join(threads[i], NULL);
  }

  int compact_offset = 0;
  for (int tree_idx = 0; tree_idx < num_trees; tree_idx++) {
    int count = counts[tree_idx];
    if (count <= 0) continue;
    int segment_start = tree_idx * leaves_per_tree;
    if (segment_start != compact_offset) {
      memmove(out_features + (size_t)compact_offset * CMCTS_FEATURE_STRIDE,
              out_features + (size_t)segment_start * CMCTS_FEATURE_STRIDE,
              (size_t)count * CMCTS_FEATURE_STRIDE * sizeof(float));
    }
    compact_offset += count;
  }

  free(counts);
  free(threads);
  free(args);
  return compact_offset;
#endif
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
      PendingEval *pending = &tree->pending_leaves[j];
      revert_virtual_loss(pending->path, pending->path_len, tree->virtual_loss);
      node_expand(tree, pending->node, &pending->state, priors + (size_t)offset * CMCTS_ACTION_SIZE);
      backup(pending->path, pending->path_len, values[offset]);
      offset += 1;
    }
    tree_clear_pending_leaves(tree);
  }
}
