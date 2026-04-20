#include "internal.h"

#include <string.h>

static int in_bounds(const CmctsState *state, int row, int col) {
  return row >= 0 && row < state->board_size && col >= 0 && col < state->board_size;
}

static int count_dir(const CmctsState *state, int row, int col, int dr, int dc, int player) {
  int total = 0;
  row += dr;
  col += dc;
  while (in_bounds(state, row, col) && state->board[row * state->board_size + col] == player) {
    total += 1;
    row += dr;
    col += dc;
  }
  return total;
}

static int is_winning_move(const CmctsState *state, int row, int col, int player) {
  const int dirs[4][2] = {{1, 0}, {0, 1}, {1, 1}, {1, -1}};
  for (int i = 0; i < 4; i++) {
    int count = 1 + count_dir(state, row, col, dirs[i][0], dirs[i][1], player) +
                count_dir(state, row, col, -dirs[i][0], -dirs[i][1], player);
    if (state->exactly_five) {
      if (count == 5) return 1;
    } else if (count >= 5) {
      return 1;
    }
  }
  return 0;
}

int state_init(CmctsState *state, int board_size, const int8_t *board, int to_play, int last_action,
               int move_count, int winner, int terminal, int exactly_five) {
  if (!state || !board || board_size < CMCTS_MIN_BOARD_SIZE || board_size > CMCTS_MAX_BOARD_SIZE) return 0;
  int action_size = board_size * board_size;
  memset(state->board, 0, sizeof(state->board));
  memcpy(state->board, board, (size_t)action_size * sizeof(int8_t));
  state->board_size = board_size;
  state->action_size = action_size;
  state->feature_stride = CMCTS_FEATURE_PLANES * action_size;
  state->to_play = to_play;
  state->last_action = last_action;
  state->move_count = move_count;
  state->winner = winner;
  state->terminal = terminal;
  state->exactly_five = exactly_five;
  return 1;
}

int state_apply_action(CmctsState *state, int action) {
  if (state->terminal || action < 0 || action >= state->action_size) return 0;
  if (state->board[action] != 0) return 0;
  int player = state->to_play;
  int row = action / state->board_size;
  int col = action % state->board_size;
  state->board[action] = (int8_t)player;
  state->last_action = action;
  state->move_count += 1;
  state->to_play = -player;
  if (is_winning_move(state, row, col, player)) {
    state->winner = player;
    state->terminal = 1;
  } else if (state->move_count == state->action_size) {
    state->winner = 0;
    state->terminal = 1;
  }
  return 1;
}

int state_legal_count(const CmctsState *state) {
  if (state->terminal) return 0;
  int count = 0;
  for (int i = 0; i < state->action_size; i++) {
    if (state->board[i] == 0) count += 1;
  }
  return count;
}

void state_write_features(const CmctsState *state, float *out) {
  float color = state->to_play == 1 ? 1.0f : 0.0f;
  for (int i = 0; i < state->action_size; i++) {
    out[i] = state->board[i] == state->to_play ? 1.0f : 0.0f;
    out[state->action_size + i] = state->board[i] == -state->to_play ? 1.0f : 0.0f;
    out[2 * state->action_size + i] = i == state->last_action ? 1.0f : 0.0f;
    out[3 * state->action_size + i] = color;
  }
}

float state_outcome_for_player(const CmctsState *state, int player) {
  if (state->winner == 0) return 0.0f;
  return state->winner == player ? 1.0f : -1.0f;
}
