#ifndef COOLRL_OMOK_CMCTS_INTERNAL_H
#define COOLRL_OMOK_CMCTS_INTERNAL_H

#include "../include/mcts.h"

#include <stddef.h>

typedef struct {
  int board_size;
  int action_size;
  int feature_stride;
  int8_t board[CMCTS_MAX_ACTION_SIZE];
  int to_play;
  int last_action;
  int move_count;
  int winner;
  int terminal;
  int exactly_five;
} CmctsState;

typedef struct Node {
  int action_size;
  int to_play;
  float prior;
  int visit_count;
  float value_sum;
  int expanded;
  struct Node **children;
} Node;

typedef struct NodeBlock NodeBlock;

typedef struct {
  CmctsState state;
  Node *node;
  Node *path[CMCTS_MAX_ACTION_SIZE + 1];
  int path_len;
} PendingEval;

struct MctsTree {
  float c_puct;
  float virtual_loss;
  int exactly_five;
  int board_size;
  int action_size;
  int feature_stride;
  CmctsState state;
  Node *root;
  float root_value;
  PendingEval *pending_roots;
  int pending_root_count;
  int pending_root_capacity;
  PendingEval *pending_leaves;
  int pending_leaf_count;
  int pending_leaf_capacity;
  NodeBlock *node_blocks;
  NodeBlock *active_node_block;
  int next_node_block_capacity;
};

int state_init(CmctsState *state, int board_size, const int8_t *board, int to_play, int last_action,
               int move_count, int winner, int terminal, int exactly_five);
int state_apply_action(CmctsState *state, int action);
int state_legal_count(const CmctsState *state);
void state_write_features(const CmctsState *state, float *out);
float state_outcome_for_player(const CmctsState *state, int player);

Node *tree_node_new(MctsTree *tree, int to_play, float prior);
void tree_reset_nodes(MctsTree *tree);
void tree_free_nodes(MctsTree *tree);
int node_child_count(const Node *node);
int node_legal_actions(const Node *node, int32_t *out_actions);
Node *node_select_child(const Node *node, float c_puct, int *out_action);
void node_expand(MctsTree *tree, Node *node, const CmctsState *state, const float *priors);
void backup(Node **path, int path_len, float value);
void apply_virtual_loss(Node **path, int path_len, float virtual_loss);
void revert_virtual_loss(Node **path, int path_len, float virtual_loss);

void tree_clear_pending_roots(MctsTree *tree);
void tree_clear_pending_leaves(MctsTree *tree);
int tree_push_pending_root(MctsTree *tree, const CmctsState *state, Node *node);
int tree_push_pending_leaf(MctsTree *tree, const CmctsState *state, Node *node,
                           Node **path, int path_len);

#endif
