#pragma once

#define BLOCK_SIZE_SWITCH(COND, CONST_NAME, ...)     \
  [&] {                                         \
    constexpr static int __base = 1024 * 256;   \
    if (COND == 1) {                           \
      constexpr static int CONST_NAME = __base * 1;     \
      return __VA_ARGS__();                     \
    } else if (COND == 2) {                    \
      constexpr static int CONST_NAME = __base * 2;     \
      return __VA_ARGS__();                     \
    } else if (COND == 3) {                   \
      constexpr static int CONST_NAME = __base * 3;    \
      return __VA_ARGS__();                     \
    } else if (COND == 4) {                   \
      constexpr static int CONST_NAME = __base * 4;    \
      return __VA_ARGS__();                     \
    } else if (COND == 5) {                   \
      constexpr static int CONST_NAME = __base * 5;    \
      return __VA_ARGS__();                     \
    } else if (COND == 6) {                   \
      constexpr static int CONST_NAME = __base * 6;    \
      return __VA_ARGS__();                     \
    } else if (COND == 7) {                   \
      constexpr static int CONST_NAME = __base * 7;    \
      return __VA_ARGS__();                     \
    } else if (COND == 8) {                   \
      constexpr static int CONST_NAME = __base * 8;    \
      return __VA_ARGS__();                     \
    } else if (COND == 16) {                   \
      constexpr static int CONST_NAME = __base * 16;    \
      return __VA_ARGS__();                     \
    } else if (COND == 32) {                   \
      constexpr static int CONST_NAME = __base * 32;    \
      return __VA_ARGS__();                     \
    } else if (COND == 36) {                   \
      constexpr static int CONST_NAME = __base * 36;    \
      return __VA_ARGS__();                     \
    } else if (COND == 40) {                   \
      constexpr static int CONST_NAME = __base * 40;    \
      return __VA_ARGS__();                     \
    } else if (COND == 44) {                   \
      constexpr static int CONST_NAME = __base * 44;    \
      return __VA_ARGS__();                     \
    } else if (COND == 48) {                   \
      constexpr static int CONST_NAME = __base * 48;    \
      return __VA_ARGS__();                     \
    } else if (COND == 52) {                   \
      constexpr static int CONST_NAME = __base * 52;    \
      return __VA_ARGS__();                     \
    } else if (COND == 56) {                   \
      constexpr static int CONST_NAME = __base * 56;    \
      return __VA_ARGS__();                     \
    } else if (COND == 60) {                   \
      constexpr static int CONST_NAME = __base * 60;    \
      return __VA_ARGS__();                     \
    } else if (COND == 64) {                   \
      constexpr static int CONST_NAME = __base * 64;    \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static int CONST_NAME = __base * 8;     \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define FP16_SWITCH(COND, ...)               \
  [&] {                                      \
    if (COND) {                              \
      using elem_type = at::Half;     \
      return __VA_ARGS__();                  \
    } else {                                 \
      using elem_type = at::BFloat16; \
      return __VA_ARGS__();                  \
    }                                        \
  }()


