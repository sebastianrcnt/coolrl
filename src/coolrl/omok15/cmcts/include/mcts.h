#ifndef COOLRL_OMOK_MCTS_H
#define COOLRL_OMOK_MCTS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CMCTS_BOARD_SIZE 15
#define CMCTS_ACTION_SIZE 225
#define CMCTS_FEATURE_PLANES 4
#define CMCTS_FEATURE_STRIDE (CMCTS_FEATURE_PLANES * CMCTS_ACTION_SIZE)

typedef struct MctsTree MctsTree;

// Threading contract: each MctsTree must be owned by one thread at a time.
// Batch APIs mutate per-tree pending evaluation queues and are not safe to call
// concurrently on the same tree.
//
// Rule contract: when exactly_five is enabled, overlines (6+ in a row) are not
// wins; play continues unless an exact five is made or the board fills.

// ---- lifecycle ----

MctsTree *mcts_tree_new(float c_puct, float virtual_loss, int exactly_five);
void mcts_tree_free(MctsTree *tree);

// Reset tree and set the root position. `board` points to a row-major int8 buffer of size 225
// with values in {-1, 0, 1}. `last_action` = -1 if none. `terminal` and `winner` describe the
// current position outcome (winner in {-1, 0, 1}; 0 means draw or game ongoing).
void mcts_tree_set_initial(MctsTree *tree,
                           const int8_t *board,
                           int to_play,
                           int last_action,
                           int move_count,
                           int winner,
                           int terminal);

// Apply `action` to the internal state and re-root the tree to the matching child.
// If the chosen child does not exist (unusual), the tree resets under the new state.
// Returns 1 on success and 0 if `action` is illegal for the current state.
int mcts_tree_advance(MctsTree *tree, int action);

// ---- batch ops ----

// `trees` is an array of `num_trees` opaque pointers. Batch entry points walk that array
// internally; pointer arrays are easy to build on the Python side via ctypes.

// For every tree whose root is not yet expanded and whose state is non-terminal, append
// its feature planes to `out_features` (shape [N, 4, 15, 15], float32, contiguous).
// `max_entries` caps N. Returns the number of entries written.
int mcts_batch_prepare_roots(MctsTree *const *trees,
                             int num_trees,
                             float *out_features,
                             int max_entries);

// Feed evaluator output for `mcts_batch_prepare_roots`. `priors` has shape [N, 225], `values`
// shape [N], where N matches the return value of the preceding prepare call.
void mcts_batch_feed_roots(MctsTree *const *trees,
                           int num_trees,
                           const float *priors,
                           const float *values);

// For each tree, apply Dirichlet noise to the root priors. `offsets` has length num_trees+1.
// noise[offsets[g]+i] corresponds to the i-th legal action (ascending action id) at the root
// of trees[g]. `epsilon` is the mixing factor. Terminal trees are skipped.
void mcts_batch_apply_root_noise(MctsTree *const *trees,
                                 int num_trees,
                                 const float *noise,
                                 const int32_t *offsets,
                                 float epsilon);

// For each tree, return the number of legal moves at the root (0 for terminal trees).
void mcts_batch_root_num_legal(MctsTree *const *trees, int num_trees, int32_t *out_counts);

// For each tree, write the root value estimate (NN value for fresh roots, running average
// for reused roots, 0 for terminal). Shape [num_trees], float32.
void mcts_batch_get_root_values(MctsTree *const *trees, int num_trees, float *out_values);

// Run one simulation round. For each non-terminal tree, attempt `leaves_per_tree` simulations;
// terminal-path backups are resolved inline. Pending leaves (non-terminal, requiring NN eval)
// are appended to `out_features`. Returns the number of pending leaves written.
int mcts_batch_collect_leaves(MctsTree *const *trees,
                              int num_trees,
                              int leaves_per_tree,
                              float *out_features,
                              int max_entries);

// Threaded variant of `mcts_batch_collect_leaves`. Work is split by tree, so a
// single MctsTree is still owned by only one thread during the call. Pending
// leaves and output features are ordered by tree index, matching
// `mcts_batch_feed_leaves`.
int mcts_batch_collect_leaves_threaded(MctsTree *const *trees,
                                       int num_trees,
                                       int leaves_per_tree,
                                       float *out_features,
                                       int max_entries,
                                       int num_threads);

// Feed evaluator output for `mcts_batch_collect_leaves`, in the same order.
void mcts_batch_feed_leaves(MctsTree *const *trees,
                            int num_trees,
                            const float *priors,
                            const float *values);

// For each tree, write root visit counts to `out_counts` (shape [num_trees, 225], float32).
// For terminal roots, all zeros.
void mcts_batch_extract_visit_counts(MctsTree *const *trees, int num_trees, float *out_counts);

#ifdef __cplusplus
}
#endif

#endif // COOLRL_OMOK_MCTS_H
