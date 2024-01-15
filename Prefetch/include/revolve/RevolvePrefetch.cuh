#pragma once

#include <cstdio>
#include <cuda_runtime.h>
#include <iostream>

#include "../Prefetch.cuh"
#include "revolve.h"

/*
 * RevolvePrefetch: A subclass of Prefetch for Revolve-specific prefetching.
 * Author: Thiago Maltempi <tmaltempi@ic.unicamp.br>
 * Author: Sandro Rigo <srigo@ic.unicamp.br>
 * Date: November 25, 2023
 */
class RevolvePrefetch : public Prefetch {

public:
  /*
   * Constructor for RevolvePrefetch.
   * @param numSnaps: Number of snapshots.
   * @param timesteps: Number of timesteps.
   * @param max_len: Maximum length.
   * @param info: Additional information (default is 0).
   */
  RevolvePrefetch(int numSnaps, int timesteps, size_t max_len, int info = 0)
      : Prefetch(numSnaps, timesteps, max_len, info, 2) {}

  /*
   * Setup method for RevolvePrefetch. Overrides the base class method.
   */
  void setup() override {
    Prefetch::setup();

    int it_revolve = -1; // Stores revolve iteration number
    int i = 0;           // Iterates over the prefetch vectors
    int check = -1; // revolve use -1 for initialization. Later, it records the
                    // number of checkpoints stored at any given time
    enum action whatodo;
    int bf[2]; // Last timesteps saved in local buffer.

    // Last iteration to have a hit in the top of local buffer.
    int last_top_hit_it = -1;
    // Last iteration to have a hit in the top of local buffer.
    int last_bot_hit_it = -1;
    // Local buffer's position of the last hit.
    int prefetch_it = -1;
    int fine = timesteps;
    int bftop = 0;
    int capo = 0;

    if (info > 2) {
      fprintf(stderr, "Pre-running Revolve\n");
      fprintf(stderr, "Range: %d to %d, Snapshots used: %d\n", capo, fine,
              snaps);
    }

    // Fill up the action vector for prefetching computation
    do {
      it_revolve++;
      int temp_info = info;
      whatodo = revolve(&check, &capo, &fine, snaps, &temp_info);

      if (whatodo == takeshot) {
        bftop = (bftop + 1) % 2;

        // saving the timestep
        bf[bftop] = capo;
      }

      if (whatodo == restore) {
        // is the buffer hit at the top position?
        if (bf[bftop] == capo) {
          last_top_hit_it = it_revolve;
          if (info > 2) {
            fprintf(stderr, "[Prefetch] Top hit: %d \n", last_top_hit_it);
          }
          continue;
        }
        // Is the buffer hit at the bottom position?
        else if (bf[(bftop + 1) % 2] == capo) {
          // Adjust top, set prefetch info and continue.
          bftop = (bftop + 1) % 2;
          last_bot_hit_it = it_revolve;
          if (prefetch_it < last_top_hit_it + 1) {
            prefetch_it = last_top_hit_it + 1;
          } else {
            prefetch_it = last_bot_hit_it + 1;
          }

          if (info > 2) {
            fprintf(stderr,
                    "[Prefetch] Bottom hit in %d. New prefecth_it: %d \n",
                    it_revolve, prefetch_it);
          }

          continue;
        }

        // Now ... deal with a miss in the local buffer. Setup the prefetch
        // If we already have a prefetch set up in the target iteration, ignore.
        if (prefetch_action.iter[i - 1] != prefetch_it) {
          if (info >= 1) {
            fprintf(
                stderr,
                "===> Predicted Buffer Hit at: %d. Setting prefetch at %d for "
                "timestep %d\n",
                it_revolve, prefetch_it, capo);
          }

          bf[(bftop + 1) % 2] = capo;
          prefetch_action.iter[i] = prefetch_it;
          prefetch_action.timestep[i] = capo;
          i++;
        } else {
          if (info >= 1) {
            fprintf(stderr,
                    "===> Predicted Buffer Miss. Conflict at iteration %d. "
                    "Skipping "
                    "prefetch for "
                    "timestep %d.\n",
                    prefetch_it, capo);
          }
        }
      }

      if (whatodo == revolve_terminate) {
        fprintf(stderr, "<TERMINATE>\n");
        fprintf(stderr, "Total iterations: %d\n", it_revolve);
        fprintf(stderr, "[Prefetch] Total prefetches defined: %d\n", i);
      }
    } while ((whatodo != revolve_terminate) && (whatodo != error));

    if (whatodo == revolve_terminate) {
      // mark the end of the prefetch vectors;
      if (i < 2000) {
        prefetch_action.timestep[i] = -1;
        prefetch_action.iter[i] = -1;
      } else {
        fprintf(stderr,
                "\nError: Prefetch data vector is too small. Max size is: %d. "
                "Positions needed: %d\n",
                2000, i);
        exit(0);
      }
    }
  }
};
