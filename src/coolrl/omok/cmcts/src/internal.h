#ifndef COOLRL_OMOK_CMCTS_INTERNAL_H
#define COOLRL_OMOK_CMCTS_INTERNAL_H

#include "../include/mcts.h"

#include <stddef.h>

typedef struct {
  int8_t board[CMCTS_ACTION_SIZE];
  int to_play;
  int last_action;
  int move_count;
  int winner;
  int terminal;
  int exactly_five;
} CmctsState;

typedef struct Node {
  int to_play;
  float prior;
  int visit_count;
  float value_sum;
  int expanded;
  struct Node *children[CMCTS_ACTION_SIZE];
} Node;

typedef struct {
  CmctsState state;
  Node *node;
  Node *path[CMCTS_ACTION_SIZE + 1];
  int path_len;
} PendingLeaf;

struct MctsTree {
  float c_puct;
  int exactly_five;
  CmctsState state;
  Node *root;
  float root_value;
  PendingLeaf *pending_roots;
  int pending_root_count;
  int pending_root_capacity;
  PendingLeaf *pending_leaves;
  int pending_leaf_count;
  int pending_leaf_capacity;
};

void state_init(CmctsState *state, const int8_t *board, int to_play, int last_action,
                int move_count, int winner, int terminal, int exactly_five);
void state_apply_action(CmctsState *state, int action);
int state_legal_count(const CmctsState *state);
void state_write_features(const CmctsState *state, float *out);
float state_outcome_for_player(const CmctsState *state, int player);

Node *node_new(int to_play, float prior);
void node_free(Node *node);
int node_child_count(const Node *node);
int node_legal_actions(const Node *node, int32_t *out_actions);
Node *node_select_child(const Node *node, float c_puct, int *out_action);
void node_expand(Node *node, const CmctsState *state, const float *priors);
void backup(Node **path, int path_len, float value);

void tree_clear_pending_roots(MctsTree *tree);
void tree_clear_pending_leaves(MctsTree *tree);
int tree_push_pending_root(MctsTree *tree, const CmctsState *state, Node *node);
int tree_push_pending_leaf(MctsTree *tree, const CmctsState *state, Node *node,
                           Node **path, int path_len);

#endif
