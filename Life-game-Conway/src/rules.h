#ifndef RULES_H
#define RULES_H

// game rules B368 / S012345678:
// - a dead cell is born if alive neighbor count is 3, 6, or 8
// - a live cell survives with ANY count 0..8
inline bool birth_rule(int n) {
  return (n == 3 || n == 6 || n == 8);
}

inline bool survive_rule(int n) {
  return (n >= 0 && n <= 8);
}

#endif // RULES_H
