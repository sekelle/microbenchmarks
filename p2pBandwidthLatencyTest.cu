/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cstdio>
#include <vector>
#include <string>

//#include <helper_cuda.h>
//#include <helper_timer.h>

using namespace std;

inline int stringRemoveDelimiter(char delimiter, const char *string) {
  int string_start = 0;

  while (string[string_start] == delimiter) {
    string_start++;
  }

  if (string_start >= static_cast<int>(strlen(string) - 1)) {
    return 0;
  }

  return string_start;
}


inline bool checkCmdLineFlag(const int argc, const char **argv,
                             const char *string_ref) {
  bool bFound = false;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];

      const char *equal_pos = strchr(string_argv, '=');
      int argv_length = static_cast<int>(
          equal_pos == 0 ? strlen(string_argv) : equal_pos - string_argv);

      int length = static_cast<int>(strlen(string_ref));

      if (length == argv_length &&
          !strncasecmp(string_argv, string_ref, length)) {
        bFound = true;
        continue;
      }
    }
  }

  return bFound;
}

inline int getCmdLineArgumentInt(const int argc, const char **argv,
                                 const char *string_ref) {
  bool bFound = false;
  int value = -1;

  if (argc >= 1) {
    for (int i = 1; i < argc; i++) {
      int string_start = stringRemoveDelimiter('-', argv[i]);
      const char *string_argv = &argv[i][string_start];
      int length = static_cast<int>(strlen(string_ref));

      if (!strncasecmp(string_argv, string_ref, length)) {
        if (length + 1 <= static_cast<int>(strlen(string_argv))) {
          int auto_inc = (string_argv[length] == '=') ? 1 : 0;
          value = atoi(&string_argv[length + auto_inc]);
        } else {
          value = 0;
        }

        bFound = true;
        continue;
      }
    }
  }

  if (bFound) {
    return value;
  } else {
    return 0;
  }
}


// Helper Timing Functions
#ifndef COMMON_HELPER_TIMER_H_
#define COMMON_HELPER_TIMER_H_

//#ifndef EXIT_WAIVED
//#define EXIT_WAIVED 2
//#endif

// Definition of the StopWatch Interface, this is used if we don't want to use
// the CUT functions But rather in a self contained class interface
class StopWatchInterface {
 public:
  StopWatchInterface() {}
  virtual ~StopWatchInterface() {}

 public:
  //! Start time measurement
  virtual void start() = 0;

  //! Stop time measurement
  virtual void stop() = 0;

  //! Reset time counters to zero
  virtual void reset() = 0;

  //! Time in msec. after start. If the stop watch is still running (i.e. there
  //! was no call to stop()) then the elapsed time is returned, otherwise the
  //! time between the last start() and stop call is returned
  virtual float getTime() = 0;

  //! Mean time to date based on the number of times the stopwatch has been
  //! _stopped_ (ie finished sessions) and the current total time
  virtual float getAverageTime() = 0;
};

//////////////////////////////////////////////////////////////////
// Begin Stopwatch timer class definitions for all OS platforms //
//////////////////////////////////////////////////////////////////
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
// includes, system
#define WINDOWS_LEAN_AND_MEAN
#include <windows.h>
#undef min
#undef max

//! Windows specific implementation of StopWatch
class StopWatchWin : public StopWatchInterface {
 public:
  //! Constructor, default
  StopWatchWin()
      : start_time(),
        end_time(),
        diff_time(0.0f),
        total_time(0.0f),
        running(false),
        clock_sessions(0),
        freq(0),
        freq_set(false) {
    if (!freq_set) {
      // helper variable
      LARGE_INTEGER temp;

      // get the tick frequency from the OS
      QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER *>(&temp));

      // convert to type in which it is needed
      freq = (static_cast<double>(temp.QuadPart)) / 1000.0;

      // rememeber query
      freq_set = true;
    }
  }

  // Destructor
  ~StopWatchWin() {}

 public:
  //! Start time measurement
  inline void start();

  //! Stop time measurement
  inline void stop();

  //! Reset time counters to zero
  inline void reset();

  //! Time in msec. after start. If the stop watch is still running (i.e. there
  //! was no call to stop()) then the elapsed time is returned, otherwise the
  //! time between the last start() and stop call is returned
  inline float getTime();

  //! Mean time to date based on the number of times the stopwatch has been
  //! _stopped_ (ie finished sessions) and the current total time
  inline float getAverageTime();

 private:
  // member variables

  //! Start of measurement
  LARGE_INTEGER start_time;
  //! End of measurement
  LARGE_INTEGER end_time;

  //! Time difference between the last start and stop
  float diff_time;

  //! TOTAL time difference between starts and stops
  float total_time;

  //! flag if the stop watch is running
  bool running;

  //! Number of times clock has been started
  //! and stopped to allow averaging
  int clock_sessions;

  //! tick frequency
  double freq;

  //! flag if the frequency has been set
  bool freq_set;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchWin::start() {
  QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&start_time));
  running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchWin::stop() {
  QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&end_time));
  diff_time = static_cast<float>(((static_cast<double>(end_time.QuadPart) -
                                   static_cast<double>(start_time.QuadPart)) /
                                  freq));

  total_time += diff_time;
  clock_sessions++;
  running = false;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchWin::reset() {
  diff_time = 0;
  total_time = 0;
  clock_sessions = 0;

  if (running) {
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&start_time));
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline float StopWatchWin::getTime() {
  // Return the TOTAL time to date
  float retval = total_time;

  if (running) {
    LARGE_INTEGER temp;
    QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER *>(&temp));
    retval += static_cast<float>(((static_cast<double>(temp.QuadPart) -
                                   static_cast<double>(start_time.QuadPart)) /
                                  freq));
  }

  return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline float StopWatchWin::getAverageTime() {
  return (clock_sessions > 0) ? (total_time / clock_sessions) : 0.0f;
}
#else
// Declarations for Stopwatch on Linux and Mac OSX
// includes, system
#include <sys/time.h>
#include <ctime>

//! Windows specific implementation of StopWatch
class StopWatchLinux : public StopWatchInterface {
 public:
  //! Constructor, default
  StopWatchLinux()
      : start_time(),
        diff_time(0.0),
        total_time(0.0),
        running(false),
        clock_sessions(0) {}

  // Destructor
  virtual ~StopWatchLinux() {}

 public:
  //! Start time measurement
  inline void start();

  //! Stop time measurement
  inline void stop();

  //! Reset time counters to zero
  inline void reset();

  //! Time in msec. after start. If the stop watch is still running (i.e. there
  //! was no call to stop()) then the elapsed time is returned, otherwise the
  //! time between the last start() and stop call is returned
  inline float getTime();

  //! Mean time to date based on the number of times the stopwatch has been
  //! _stopped_ (ie finished sessions) and the current total time
  inline float getAverageTime();

 private:
  // helper functions

  //! Get difference between start time and current time
  inline float getDiffTime();

 private:
  // member variables

  //! Start of measurement
  struct timeval start_time;

  //! Time difference between the last start and stop
  float diff_time;

  //! TOTAL time difference between starts and stops
  float total_time;

  //! flag if the stop watch is running
  bool running;

  //! Number of times clock has been started
  //! and stopped to allow averaging
  int clock_sessions;
};

// functions, inlined

////////////////////////////////////////////////////////////////////////////////
//! Start time measurement
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchLinux::start() {
  gettimeofday(&start_time, 0);
  running = true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop time measurement and increment add to the current diff_time summation
//! variable. Also increment the number of times this clock has been run.
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchLinux::stop() {
  diff_time = getDiffTime();
  total_time += diff_time;
  running = false;
  clock_sessions++;
}

////////////////////////////////////////////////////////////////////////////////
//! Reset the timer to 0. Does not change the timer running state but does
//! recapture this point in time as the current start time if it is running.
////////////////////////////////////////////////////////////////////////////////
inline void StopWatchLinux::reset() {
  diff_time = 0;
  total_time = 0;
  clock_sessions = 0;

  if (running) {
    gettimeofday(&start_time, 0);
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. after start. If the stop watch is still running (i.e. there
//! was no call to stop()) then the elapsed time is returned added to the
//! current diff_time sum, otherwise the current summed time difference alone
//! is returned.
////////////////////////////////////////////////////////////////////////////////
inline float StopWatchLinux::getTime() {
  // Return the TOTAL time to date
  float retval = total_time;

  if (running) {
    retval += getDiffTime();
  }

  return retval;
}

////////////////////////////////////////////////////////////////////////////////
//! Time in msec. for a single run based on the total number of COMPLETED runs
//! and the total time.
////////////////////////////////////////////////////////////////////////////////
inline float StopWatchLinux::getAverageTime() {
  return (clock_sessions > 0) ? (total_time / clock_sessions) : 0.0f;
}
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
inline float StopWatchLinux::getDiffTime() {
  struct timeval t_time;
  gettimeofday(&t_time, 0);

  // time difference in milli-seconds
  return static_cast<float>(1000.0 * (t_time.tv_sec - start_time.tv_sec) +
                            (0.001 * (t_time.tv_usec - start_time.tv_usec)));
}
#endif  // WIN32

////////////////////////////////////////////////////////////////////////////////
//! Timer functionality exported

////////////////////////////////////////////////////////////////////////////////
//! Create a new timer
//! @return true if a time has been created, otherwise false
//! @param  name of the new timer, 0 if the creation failed
////////////////////////////////////////////////////////////////////////////////
inline bool sdkCreateTimer(StopWatchInterface **timer_interface) {
// printf("sdkCreateTimer called object %08x\n", (void *)*timer_interface);
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
  *timer_interface = reinterpret_cast<StopWatchInterface *>(new StopWatchWin());
#else
  *timer_interface =
      reinterpret_cast<StopWatchInterface *>(new StopWatchLinux());
#endif
  return (*timer_interface != NULL) ? true : false;
}

////////////////////////////////////////////////////////////////////////////////
//! Delete a timer
//! @return true if a time has been deleted, otherwise false
//! @param  name of the timer to delete
////////////////////////////////////////////////////////////////////////////////
inline bool sdkDeleteTimer(StopWatchInterface **timer_interface) {
  // printf("sdkDeleteTimer called object %08x\n", (void *)*timer_interface);
  if (*timer_interface) {
    delete *timer_interface;
    *timer_interface = NULL;
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Start the time with name \a name
//! @param name  name of the timer to start
////////////////////////////////////////////////////////////////////////////////
inline bool sdkStartTimer(StopWatchInterface **timer_interface) {
  // printf("sdkStartTimer called object %08x\n", (void *)*timer_interface);
  if (*timer_interface) {
    (*timer_interface)->start();
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Stop the time with name \a name. Does not reset.
//! @param name  name of the timer to stop
////////////////////////////////////////////////////////////////////////////////
inline bool sdkStopTimer(StopWatchInterface **timer_interface) {
  // printf("sdkStopTimer called object %08x\n", (void *)*timer_interface);
  if (*timer_interface) {
    (*timer_interface)->stop();
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Resets the timer's counter.
//! @param name  name of the timer to reset.
////////////////////////////////////////////////////////////////////////////////
inline bool sdkResetTimer(StopWatchInterface **timer_interface) {
  // printf("sdkResetTimer called object %08x\n", (void *)*timer_interface);
  if (*timer_interface) {
    (*timer_interface)->reset();
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////
//! Return the average time for timer execution as the total time
//! for the timer dividied by the number of completed (stopped) runs the timer
//! has made.
//! Excludes the current running time if the timer is currently running.
//! @param name  name of the timer to return the time of
////////////////////////////////////////////////////////////////////////////////
inline float sdkGetAverageTimerValue(StopWatchInterface **timer_interface) {
  //  printf("sdkGetAverageTimerValue called object %08x\n", (void
  //  *)*timer_interface);
  if (*timer_interface) {
    return (*timer_interface)->getAverageTime();
  } else {
    return 0.0f;
  }
}

////////////////////////////////////////////////////////////////////////////////
//! Total execution time for the timer over all runs since the last reset
//! or timer creation.
//! @param name  name of the timer to obtain the value of.
////////////////////////////////////////////////////////////////////////////////
inline float sdkGetTimerValue(StopWatchInterface **timer_interface) {
  // printf("sdkGetTimerValue called object %08x\n", (void *)*timer_interface);
  if (*timer_interface) {
    return (*timer_interface)->getTime();
  } else {
    return 0.0f;
  }
}

#endif  // COMMON_HELPER_TIMER_H_



const char *sSampleName = "P2P (Peer-to-Peer) GPU Bandwidth Latency Test";

typedef enum {
  P2P_WRITE = 0,
  P2P_READ = 1,
} P2PDataTransfer;

typedef enum {
  CE = 0,
  SM = 1,
} P2PEngine;

P2PEngine p2p_mechanism = CE;  // By default use Copy Engine

// Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }
__global__ void delay(volatile int *flag,
                      unsigned long long timeout_clocks = 10000000) {
  // Wait until the application notifies us that it has completed queuing up the
  // experiment, or timeout and exit, allowing the application to make progress
  long long int start_clock, sample_clock;
  start_clock = clock64();

  while (!*flag) {
    sample_clock = clock64();

    if (sample_clock - start_clock > timeout_clocks) {
      break;
    }
  }
}

// This kernel is for demonstration purposes only, not a performant kernel for
// p2p transfers.
__global__ void copyp2p(int4 *__restrict__ dest, int4 const *__restrict__ src,
                        size_t num_elems) {
  size_t globalId = blockIdx.x * blockDim.x + threadIdx.x;
  size_t gridSize = blockDim.x * gridDim.x;

#pragma unroll(5)
  for (size_t i = globalId; i < num_elems; i += gridSize) {
    dest[i] = src[i];
  }
}

///////////////////////////////////////////////////////////////////////////
// Print help screen
///////////////////////////////////////////////////////////////////////////
void printHelp(void) {
  printf("Usage:  p2pBandwidthLatencyTest [OPTION]...\n");
  printf("Tests bandwidth/latency of GPU pairs using P2P and without P2P\n");
  printf("\n");

  printf("Options:\n");
  printf("--help\t\tDisplay this help menu\n");
  printf(
      "--p2p_read\tUse P2P reads for data transfers between GPU pairs and show "
      "corresponding results.\n \t\tDefault used is P2P write operation.\n");
  printf("--sm_copy                      Use SM intiated p2p transfers instead of Copy Engine\n");
  printf("--numElems=<NUM_OF_INT_ELEMS>  Number of integer elements to be used in p2p copy.\n");
}

void checkP2Paccess(int numGPUs) {
  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);
    cudaCheckError();

    for (int j = 0; j < numGPUs; j++) {
      int access;
      if (i != j) {
        cudaDeviceCanAccessPeer(&access, i, j);
        cudaCheckError();
        printf("Device=%d %s Access Peer Device=%d\n", i,
               access ? "CAN" : "CANNOT", j);
      }
    }
  }
  printf(
      "\n***NOTE: In case a device doesn't have P2P access to other one, it "
      "falls back to normal memcopy procedure.\nSo you can see lesser "
      "Bandwidth (GB/s) and unstable Latency (us) in those cases.\n\n");
}

void performP2PCopy(int *dest, int destDevice, int *src, int srcDevice,
                    int num_elems, int repeat, bool p2paccess,
                    cudaStream_t streamToRun) {
  int blockSize = 0;
  int numBlocks = 0;

  cudaOccupancyMaxPotentialBlockSize(&numBlocks, &blockSize, copyp2p);
  cudaCheckError();

  if (p2p_mechanism == SM && p2paccess) {
    for (int r = 0; r < repeat; r++) {
      copyp2p<<<numBlocks, blockSize, 0, streamToRun>>>(
          (int4 *)dest, (int4 *)src, num_elems / 4);
    }
  } else {
    for (int r = 0; r < repeat; r++) {
      cudaMemcpyPeerAsync(dest, destDevice, src, srcDevice,
                          sizeof(int) * num_elems, streamToRun);
    }
  }
}

void outputBandwidthMatrix(int numElems, int numGPUs, bool p2p, P2PDataTransfer p2p_method) {
  int repeat = 5;
  volatile int *flag = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);  // buffer for D2D, that is, intra-GPU copy
  vector<cudaEvent_t> start(numGPUs);
  vector<cudaEvent_t> stop(numGPUs);
  vector<cudaStream_t> stream(numGPUs);

  cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable);
  cudaCheckError();

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking);
    cudaMalloc(&buffers[d], numElems * sizeof(int));
    cudaCheckError();
    cudaMemset(buffers[d], 0, numElems * sizeof(int));
    cudaCheckError();
    cudaMalloc(&buffersD2D[d], numElems * sizeof(int));
    cudaCheckError();
    cudaMemset(buffersD2D[d], 0, numElems * sizeof(int));
    cudaCheckError();
    cudaEventCreate(&start[d]);
    cudaCheckError();
    cudaEventCreate(&stop[d]);
    cudaCheckError();
  }

  vector<double> bandwidthMatrix(numGPUs * numGPUs);

  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        cudaDeviceCanAccessPeer(&access, i, j);
        if (access) {
          cudaDeviceEnablePeerAccess(j, 0);
          cudaCheckError();
          cudaSetDevice(j);
          cudaCheckError();
          cudaDeviceEnablePeerAccess(i, 0);
          cudaCheckError();
          cudaSetDevice(i);
          cudaCheckError();
        }
      }

      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      // Block the stream until all the work is queued up
      // DANGER! - cudaMemcpy*Async may infinitely block waiting for
      // room to push the operation, so keep the number of repeatitions
      // relatively low.  Higher repeatitions will cause the delay kernel
      // to timeout and lead to unstable results.
      *flag = 0;
      delay<<<1, 1, 0, stream[i]>>>(flag);
      cudaCheckError();
      cudaEventRecord(start[i], stream[i]);
      cudaCheckError();

      if (i == j) {
        // Perform intra-GPU, D2D copies
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat,
                       access, stream[i]);

      } else {
        if (p2p_method == P2P_WRITE) {
          performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access,
                         stream[i]);
        } else {
          performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access,
                         stream[i]);
        }
      }

      cudaEventRecord(stop[i], stream[i]);
      cudaCheckError();

      // Release the queued events
      *flag = 1;
      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      float time_ms;
      cudaEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = numElems * sizeof(int) * repeat / (double)1e9;
      if (i == j) {
        gb *= 2;  // must count both the read and the write here
      }
      bandwidthMatrix[i * numGPUs + j] = gb / time_s;
      if (p2p && access) {
        cudaDeviceDisablePeerAccess(j);
        cudaSetDevice(j);
        cudaDeviceDisablePeerAccess(i);
        cudaSetDevice(i);
        cudaCheckError();
      }
    }
  }

  printf("   D\\D");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d ", j);
  }

  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d ", i);

    for (int j = 0; j < numGPUs; j++) {
      printf("%6.02f ", bandwidthMatrix[i * numGPUs + j]);
    }

    printf("\n");
  }

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaFree(buffers[d]);
    cudaFree(buffersD2D[d]);
    cudaCheckError();
    cudaEventDestroy(start[d]);
    cudaCheckError();
    cudaEventDestroy(stop[d]);
    cudaCheckError();
    cudaStreamDestroy(stream[d]);
    cudaCheckError();
  }

  cudaFreeHost((void *)flag);
  cudaCheckError();
}

void outputBidirectionalBandwidthMatrix(int numElems, int numGPUs, bool p2p) {
  int repeat = 5;
  volatile int *flag = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);
  vector<cudaEvent_t> start(numGPUs);
  vector<cudaEvent_t> stop(numGPUs);
  vector<cudaStream_t> stream0(numGPUs);
  vector<cudaStream_t> stream1(numGPUs);

  cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable);
  cudaCheckError();

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaMalloc(&buffers[d], numElems * sizeof(int));
    cudaMemset(buffers[d], 0, numElems * sizeof(int));
    cudaMalloc(&buffersD2D[d], numElems * sizeof(int));
    cudaMemset(buffersD2D[d], 0, numElems * sizeof(int));
    cudaCheckError();
    cudaEventCreate(&start[d]);
    cudaCheckError();
    cudaEventCreate(&stop[d]);
    cudaCheckError();
    cudaStreamCreateWithFlags(&stream0[d], cudaStreamNonBlocking);
    cudaCheckError();
    cudaStreamCreateWithFlags(&stream1[d], cudaStreamNonBlocking);
    cudaCheckError();
  }

  vector<double> bandwidthMatrix(numGPUs * numGPUs);

  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        cudaDeviceCanAccessPeer(&access, i, j);
        if (access) {
          cudaSetDevice(i);
          cudaDeviceEnablePeerAccess(j, 0);
          cudaCheckError();
          cudaSetDevice(j);
          cudaDeviceEnablePeerAccess(i, 0);
          cudaCheckError();
        }
      }

      cudaSetDevice(i);
      cudaStreamSynchronize(stream0[i]);
      cudaStreamSynchronize(stream1[j]);
      cudaCheckError();

      // Block the stream until all the work is queued up
      // DANGER! - cudaMemcpy*Async may infinitely block waiting for
      // room to push the operation, so keep the number of repeatitions
      // relatively low.  Higher repeatitions will cause the delay kernel
      // to timeout and lead to unstable results.
      *flag = 0;
      cudaSetDevice(i);
      // No need to block stream1 since it'll be blocked on stream0's event
      delay<<<1, 1, 0, stream0[i]>>>(flag);
      cudaCheckError();

      // Force stream1 not to start until stream0 does, in order to ensure
      // the events on stream0 fully encompass the time needed for all
      // operations
      cudaEventRecord(start[i], stream0[i]);
      cudaStreamWaitEvent(stream1[j], start[i], 0);

      if (i == j) {
        // For intra-GPU perform 2 memcopies buffersD2D <-> buffers
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat,
                       access, stream0[i]);
        performP2PCopy(buffersD2D[i], i, buffers[i], i, numElems, repeat,
                       access, stream1[i]);
      } else {
        if (access && p2p_mechanism == SM) {
          cudaSetDevice(j);
        }
        performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access,
                       stream1[j]);
        if (access && p2p_mechanism == SM) {
          cudaSetDevice(i);
        }
        performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access,
                       stream0[i]);
      }

      // Notify stream0 that stream1 is complete and record the time of
      // the total transaction
      cudaEventRecord(stop[j], stream1[j]);
      cudaStreamWaitEvent(stream0[i], stop[j], 0);
      cudaEventRecord(stop[i], stream0[i]);

      // Release the queued operations
      *flag = 1;
      cudaStreamSynchronize(stream0[i]);
      cudaStreamSynchronize(stream1[j]);
      cudaCheckError();

      float time_ms;
      cudaEventElapsedTime(&time_ms, start[i], stop[i]);
      double time_s = time_ms / 1e3;

      double gb = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;
      if (i == j) {
        gb *= 2;  // must count both the read and the write here
      }
      bandwidthMatrix[i * numGPUs + j] = gb / time_s;
      if (p2p && access) {
        cudaSetDevice(i);
        cudaDeviceDisablePeerAccess(j);
        cudaSetDevice(j);
        cudaDeviceDisablePeerAccess(i);
      }
    }
  }

  printf("   D\\D");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d ", j);
  }

  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d ", i);

    for (int j = 0; j < numGPUs; j++) {
      printf("%6.02f ", bandwidthMatrix[i * numGPUs + j]);
    }

    printf("\n");
  }

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaFree(buffers[d]);
    cudaFree(buffersD2D[d]);
    cudaCheckError();
    cudaEventDestroy(start[d]);
    cudaCheckError();
    cudaEventDestroy(stop[d]);
    cudaCheckError();
    cudaStreamDestroy(stream0[d]);
    cudaCheckError();
    cudaStreamDestroy(stream1[d]);
    cudaCheckError();
  }

  cudaFreeHost((void *)flag);
  cudaCheckError();
}

void outputLatencyMatrix(int numGPUs, bool p2p, P2PDataTransfer p2p_method) {
  int repeat = 100;
  int numElems = 4;  // perform 1-int4 transfer.
  volatile int *flag = NULL;
  StopWatchInterface *stopWatch = NULL;
  vector<int *> buffers(numGPUs);
  vector<int *> buffersD2D(numGPUs);  // buffer for D2D, that is, intra-GPU copy
  vector<cudaStream_t> stream(numGPUs);
  vector<cudaEvent_t> start(numGPUs);
  vector<cudaEvent_t> stop(numGPUs);

  cudaHostAlloc((void **)&flag, sizeof(*flag), cudaHostAllocPortable);
  cudaCheckError();

  if (!sdkCreateTimer(&stopWatch)) {
    printf("Failed to create stop watch\n");
    exit(EXIT_FAILURE);
  }
  sdkStartTimer(&stopWatch);

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking);
    cudaMalloc(&buffers[d], sizeof(int) * numElems);
    cudaMemset(buffers[d], 0, sizeof(int) * numElems);
    cudaMalloc(&buffersD2D[d], sizeof(int) * numElems);
    cudaMemset(buffersD2D[d], 0, sizeof(int) * numElems);
    cudaCheckError();
    cudaEventCreate(&start[d]);
    cudaCheckError();
    cudaEventCreate(&stop[d]);
    cudaCheckError();
  }

  vector<double> gpuLatencyMatrix(numGPUs * numGPUs);
  vector<double> cpuLatencyMatrix(numGPUs * numGPUs);

  for (int i = 0; i < numGPUs; i++) {
    cudaSetDevice(i);

    for (int j = 0; j < numGPUs; j++) {
      int access = 0;
      if (p2p) {
        cudaDeviceCanAccessPeer(&access, i, j);
        if (access) {
          cudaDeviceEnablePeerAccess(j, 0);
          cudaCheckError();
          cudaSetDevice(j);
          cudaDeviceEnablePeerAccess(i, 0);
          cudaSetDevice(i);
          cudaCheckError();
        }
      }
      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      // Block the stream until all the work is queued up
      // DANGER! - cudaMemcpy*Async may infinitely block waiting for
      // room to push the operation, so keep the number of repeatitions
      // relatively low.  Higher repeatitions will cause the delay kernel
      // to timeout and lead to unstable results.
      *flag = 0;
      delay<<<1, 1, 0, stream[i]>>>(flag);
      cudaCheckError();
      cudaEventRecord(start[i], stream[i]);

      sdkResetTimer(&stopWatch);
      if (i == j) {
        // Perform intra-GPU, D2D copies
        performP2PCopy(buffers[i], i, buffersD2D[i], i, numElems, repeat,
                       access, stream[i]);
      } else {
        if (p2p_method == P2P_WRITE) {
          performP2PCopy(buffers[j], j, buffers[i], i, numElems, repeat, access,
                         stream[i]);
        } else {
          performP2PCopy(buffers[i], i, buffers[j], j, numElems, repeat, access,
                         stream[i]);
        }
      }
      float cpu_time_ms = sdkGetTimerValue(&stopWatch);

      cudaEventRecord(stop[i], stream[i]);
      // Now that the work has been queued up, release the stream
      *flag = 1;
      cudaStreamSynchronize(stream[i]);
      cudaCheckError();

      float gpu_time_ms;
      cudaEventElapsedTime(&gpu_time_ms, start[i], stop[i]);

      gpuLatencyMatrix[i * numGPUs + j] = gpu_time_ms * 1e3 / repeat;
      cpuLatencyMatrix[i * numGPUs + j] = cpu_time_ms * 1e3 / repeat;
      if (p2p && access) {
        cudaDeviceDisablePeerAccess(j);
        cudaSetDevice(j);
        cudaDeviceDisablePeerAccess(i);
        cudaSetDevice(i);
        cudaCheckError();
      }
    }
  }

  printf("   GPU");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d ", j);
  }

  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d ", i);

    for (int j = 0; j < numGPUs; j++) {
      printf("%6.02f ", gpuLatencyMatrix[i * numGPUs + j]);
    }

    printf("\n");
  }

  printf("\n   CPU");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d ", j);
  }

  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d ", i);

    for (int j = 0; j < numGPUs; j++) {
      printf("%6.02f ", cpuLatencyMatrix[i * numGPUs + j]);
    }

    printf("\n");
  }

  for (int d = 0; d < numGPUs; d++) {
    cudaSetDevice(d);
    cudaFree(buffers[d]);
    cudaFree(buffersD2D[d]);
    cudaCheckError();
    cudaEventDestroy(start[d]);
    cudaCheckError();
    cudaEventDestroy(stop[d]);
    cudaCheckError();
    cudaStreamDestroy(stream[d]);
    cudaCheckError();
  }

  sdkDeleteTimer(&stopWatch);

  cudaFreeHost((void *)flag);
  cudaCheckError();
}

int main(int argc, char **argv) {
  int numGPUs, numElems = 40000000;
  P2PDataTransfer p2p_method = P2P_WRITE;

  cudaGetDeviceCount(&numGPUs);
  cudaCheckError();

  // process command line args
  if (checkCmdLineFlag(argc, (const char **)argv, "help")) {
    printHelp();
    return 0;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "p2p_read")) {
    p2p_method = P2P_READ;
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "sm_copy")) {
    p2p_mechanism = SM;
  }

  // number of elements of int to be used in copy.
  if (checkCmdLineFlag(argc, (const char **)argv, "numElems")) {
    numElems = getCmdLineArgumentInt(argc, (const char **)argv, "numElems");
  }

  printf("[%s]\n", sSampleName);

  // output devices
  for (int i = 0; i < numGPUs; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cudaCheckError();
    printf("Device: %d, %s, pciBusID: %x, pciDeviceID: %x, pciDomainID:%x\n", i,
           prop.name, prop.pciBusID, prop.pciDeviceID, prop.pciDomainID);
  }

  checkP2Paccess(numGPUs);

  // Check peer-to-peer connectivity
  printf("P2P Connectivity Matrix\n");
  printf("     D\\D");

  for (int j = 0; j < numGPUs; j++) {
    printf("%6d", j);
  }
  printf("\n");

  for (int i = 0; i < numGPUs; i++) {
    printf("%6d\t", i);
    for (int j = 0; j < numGPUs; j++) {
      if (i != j) {
        int access;
        cudaDeviceCanAccessPeer(&access, i, j);
        cudaCheckError();
        printf("%6d", (access) ? 1 : 0);
      } else {
        printf("%6d", 1);
      }
    }
    printf("\n");
  }

  printf("Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
  outputBandwidthMatrix(numElems, numGPUs, false, P2P_WRITE);
  printf("Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)\n");
  outputBandwidthMatrix(numElems, numGPUs, true, P2P_WRITE);
  if (p2p_method == P2P_READ) {
    printf("Unidirectional P2P=Enabled Bandwidth (P2P Reads) Matrix (GB/s)\n");
    outputBandwidthMatrix(numElems, numGPUs, true, p2p_method);
  }
  printf("Bidirectional P2P=Disabled Bandwidth Matrix (GB/s)\n");
  outputBidirectionalBandwidthMatrix(numElems, numGPUs, false);
  printf("Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)\n");
  outputBidirectionalBandwidthMatrix(numElems, numGPUs, true);

  printf("P2P=Disabled Latency Matrix (us)\n");
  outputLatencyMatrix(numGPUs, false, P2P_WRITE);
  printf("P2P=Enabled Latency (P2P Writes) Matrix (us)\n");
  outputLatencyMatrix(numGPUs, true, P2P_WRITE);
  if (p2p_method == P2P_READ) {
    printf("P2P=Enabled Latency (P2P Reads) Matrix (us)\n");
    outputLatencyMatrix(numGPUs, true, p2p_method);
  }

  printf(
      "\nNOTE: The CUDA Samples are not meant for performance measurements. "
      "Results may vary when GPU Boost is enabled.\n");

  exit(EXIT_SUCCESS);
}
