#include "internal.h"

#include <string.h>

static int in_bounds(int row, int col) {
  return row >= 0 && row < CMCTS_BOARD_SIZE && col >= 0 && col < CMCTS_BOARD_SIZE;
}

static int count_dir(const CmctsState *state, int row, int col, int dr, int dc, int player) {
  int total = 0;
  row += dr;
  col += dc;
  while (in_bounds(row, col) && state->board[row * CMCTS_BOARD_SIZE + col] == player) {
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

void state_init(CmctsState *state, const int8_t *board, int to_play, int last_action,
                int move_count, int winner, int terminal, int exactly_five) {
  memcpy(state->board, board, CMCTS_ACTION_SIZE * sizeof(int8_t));
  state->to_play = to_play;
  state->last_action = last_action;
  state->move_count = move_count;
  state->winner = winner;
  state->terminal = terminal;
  state->exactly_five = exactly_five;
}

void state_apply_action(CmctsState *state, int action) {
  if (state->terminal || action < 0 || action >= CMCTS_ACTION_SIZE) return;
  if (state->board[action] != 0) return;
  int player = state->to_play;
  int row = action / CMCTS_BOARD_SIZE;
  int col = action % CMCTS_BOARD_SIZE;
  state->board[action] = (int8_t)player;
  state->last_action = action;
  state->move_count += 1;
  state->to_play = -player;
  if (is_winning_move(state, row, col, player)) {
    state->winner = player;
    state->terminal = 1;
  } else if (state->move_count == CMCTS_ACTION_SIZE) {
    state->winner = 0;
    state->terminal = 1;
  }
}

int state_legal_count(const CmctsState *state) {
  if (state->terminal) return 0;
  int count = 0;
  for (int i = 0; i < CMCTS_ACTION_SIZE; i++) {
    if (state->board[i] == 0) count += 1;
  }
  return count;
}

void state_write_features(const CmctsState *state, float *out) {
  float color = state->to_play == 1 ? 1.0f : 0.0f;
  for (int i = 0; i < CMCTS_ACTION_SIZE; i++) {
    out[i] = state->board[i] == state->to_play ? 1.0f : 0.0f;
    out[CMCTS_ACTION_SIZE + i] = state->board[i] == -state->to_play ? 1.0f : 0.0f;
    out[2 * CMCTS_ACTION_SIZE + i] = i == state->last_action ? 1.0f : 0.0f;
    out[3 * CMCTS_ACTION_SIZE + i] = color;
  }
}

float state_outcome_for_player(const CmctsState *state, int player) {
  if (state->winner == 0) return 0.0f;
  return state->winner == player ? 1.0f : -1.0f;
}
