/*
 * Copyright (c) 2012 David Rodrigues
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *
 * NOTE
 * ----
 * This is a slightly modified version of the macro logger found at
 * https://github.com/dmcrodrigues/macro-logger/blob/master/macrologger.h
 */

#ifndef __MACROLOGGER_H__
#define __MACROLOGGER_H__

#include <time.h>
#include <string.h>
#include <unistd.h>
#include <omp.h>

// === auxiliary functions
static inline char *timenow();

#define _FILE strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__

#define NO_LOG          0x00
#define ERROR_LEVEL     0x01
#define WARNING_LEVEL   0x02
#define INFO_LEVEL      0x03
#define DEBUG_LEVEL     0x04
#define SUPER_DEBUG_LEVEL  0x05
#define ULTRA_DEBUG_LEVEL  0x06


#ifndef LOG_LEVEL
#define LOG_LEVEL   WARNING_LEVEL
#endif

#define PRINTFUNCTION(format, ...)      fprintf(stderr, format, __VA_ARGS__)
#define PRINTOUTFUNCTION(format, ...)   fprintf(stdout, format, __VA_ARGS__)


#define LOG_FMT             "%s | %-7s | %-15s | %s:%d [pid=%d/thr=%d] | "
#define LOG_ARGS(LOG_TAG)   timenow(), LOG_TAG, _FILE, __FUNCTION__, __LINE__, getpid(), omp_get_thread_num()


#define NEWLINE     "\n"

#define ERROR_TAG   "ERROR"
#define WARNING_TAG "WARNING"
#define INFO_TAG    "INFO"
#define DEBUG_TAG   "DEBUG"
#define SUPER_DEBUG_TAG   "SUPER-DEBUG"
#define ULTRA_DEBUG_TAG   "ULTRA-DEBUG"

#if LOG_LEVEL >= ULTRA_DEBUG_LEVEL
#define LOG_ULTRA_DEBUG(message, args...)     PRINTFUNCTION(LOG_FMT message NEWLINE, LOG_ARGS(ULTRA_DEBUG_TAG), ## args)
#else
#define LOG_ULTRA_DEBUG(message, args...)
#endif

#if LOG_LEVEL >= SUPER_DEBUG_LEVEL
#define LOG_SUPER_DEBUG(message, args...)     PRINTFUNCTION(LOG_FMT message NEWLINE, LOG_ARGS(SUPER_DEBUG_TAG), ## args)
#else
#define LOG_SUPER_DEBUG(message, args...)
#endif

#if LOG_LEVEL >= DEBUG_LEVEL
#define LOG_DEBUG(message, args...)     PRINTFUNCTION(LOG_FMT message NEWLINE, LOG_ARGS(DEBUG_TAG), ## args)
#else
#define LOG_DEBUG(message, args...)
#endif

#if LOG_LEVEL >= INFO_LEVEL
#define LOG_INFO(message, args...)      PRINTFUNCTION(LOG_FMT message NEWLINE, LOG_ARGS(INFO_TAG), ## args)
#else
#define LOG_INFO(message, args...)
#endif

#if LOG_LEVEL >= WARNING_LEVEL
#define LOG_WARNING(message, args...)     PRINTFUNCTION(LOG_FMT message NEWLINE, LOG_ARGS(WARNING_TAG), ## args)
#else
#define LOG_WARNING(message, args...)
#endif

#if LOG_LEVEL >= ERROR_LEVEL
#define LOG_ERROR(message, args...)     PRINTFUNCTION(LOG_FMT message NEWLINE, LOG_ARGS(ERROR_TAG), ## args)
#else
#define LOG_ERROR(message, args...)
#endif

#if LOG_LEVEL >= NO_LOGS
#define LOG_IF_ERROR(condition, message, args...) if (condition) PRINTFUNCTION(LOG_FMT message NEWLINE, LOG_ARGS(ERROR_TAG), ## args)
#else
#define LOG_IF_ERROR(condition, message, args...)
#endif

static inline char *timenow() {
    static char buffer[64];
    time_t rawtime;
    struct tm *timeinfo;

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, 64, "%Y-%m-%d %H:%M:%S", timeinfo);

    return buffer;
}

#endif
