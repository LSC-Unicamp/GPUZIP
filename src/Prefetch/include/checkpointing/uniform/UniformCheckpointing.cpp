#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <map>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#include "../Checkpointing.hpp"

#include "../../common/GPUZIPLogger.cpp"

/**
 * @class UniformCheckpointing
 * @brief Implements a checkpointing mechanism using uniform checkpoint spacing.
 * @author Bruno Ortega <brunoteixeira@estudante.ufscar.br>
 * @date Jun 3rd, 2026
 *
 * This class extends the base `Checkpointing` class to provide specific
 * checkpointing actions (save, restore, forward, backward, terminate) using
 * a fixed-spacing checkpoint distribution strategy.
 *
 * The algorithm stores checkpoints at approximately uniform timestep
 * intervals and, during the adjoint phase, restores the most recent
 * checkpoint and recomputes forward states as needed before executing
 * backward operations.
 */
class UniformCheckpointing : public Checkpointing {

private:
  std::vector<int> checkpoints; //< Vector that stores the timestep value of each checkpoint.
  int checkpoint_idx = 0;       //< Checkpoint index to access its timestep value.
  bool adjoint = false;         //< Indicates whether execution is currently in
                                // the forward or in the adjoint phase.
  bool save = false;            //< Controls the two-step checkpoint creation process:
                                // first issue FORWARD, then SAVE for the same timestep.
  bool restore = false;         //< Controls the two-step restore sequence:
                                // first execute BACKWARD at a checkpoint boundary,
                                // then issue RESTORE on the next scheduler call.
  int current_ts = 1;           //< Current timestep.
  int adj_fwd_ts = 0;           //< Current timestep during forward recomputation
                                // in the adjoint phase.

protected:

  /**
   * @brief Resets the internal state of the checkpointing process.
   *
   * Sets `checkpoints`, `checkpoint_idx`, `adjoint`, `save`, `restore`,
   * `current_ts` and `adj_fwd_ts` to their initial values.
   * This is typically called to reinitialize the checkpointing algorithm.
   */
  void reset() override {
    checkpoints.clear();
    checkpoint_idx = 0;
    adjoint = false;
    save = false;
    restore = false;
    current_ts = 1;
    adj_fwd_ts = 0;
  }

  /**
   * @brief Sets the checkpoints vector with its timesteps.
   *
   * Computes approximately uniformly spaced checkpoint locations
   * and stores their timestep indices in the internal checkpoint list.
   */
  void setCheckpoints() {

    checkpoints.push_back(1);

    for (int i = 1; i < snaps; i++) {
        int cp = std::round(i * static_cast<double>(steps) / snaps);
        checkpoints.push_back(cp);
    }
  }

  /**
   * @brief Determines the next action to perform in the checkpointing process.
   *
   * @return An `Action` object describing the next step, including its type and
   * relevant parameters.
   */
  Action getAction() override {
    // Forward from first to last timestep
    if(!adjoint){
      // At last timestep, forward finishes and adjoint begins
      if(current_ts == steps) {
        adjoint = true;
        checkpoint_idx--;
        return Action(current_ts, ACTION_FORWARD);
      }            
      
      // Apply forward and save for the current timestep
      if(current_ts == checkpoints[checkpoint_idx]) {
        if(!save) {
          save = true;
          return Action(current_ts, ACTION_FORWARD);
        }
        save = false;
        current_ts++;
        checkpoint_idx++;
        return Action(current_ts-1, ACTION_SAVE);
      }

      // Apply forward for the current timestep
      current_ts++;
      return Action(current_ts-1, ACTION_FORWARD);
    }

    // Adjoint from last to first timestep
    // Beginning of a recomputation interval.
    if(current_ts == checkpoints[checkpoint_idx+1] || current_ts == steps){
      
      // First visit: execute backward at the interval boundary.
      if(!restore) {
        restore = true;
        return Action(current_ts, ACTION_BACKWARD);
      }

      // No remaining checkpoints to restore: adjoint phase finished.
      if(checkpoint_idx < 0)
        return Action(current_ts, ACTION_TERMINATE);

      // Second visit: restore the previous checkpoint.
      adj_fwd_ts = checkpoints[checkpoint_idx];
      restore = false;
      current_ts--;
      checkpoint_idx--;
      return Action(checkpoints[checkpoint_idx+1], ACTION_RESTORE);
    }

    // Recompute forward states from the restored checkpoint
    // until reaching the current adjoint timestep.
    if(adj_fwd_ts <= current_ts) {
      adj_fwd_ts++;
      return Action(adj_fwd_ts-1, ACTION_FORWARD);
    } else { // Recomputed state available: execute backward.
      current_ts--;
      adj_fwd_ts = checkpoints[checkpoint_idx+1];
      return Action(current_ts+1, ACTION_BACKWARD);
    }

    return Action(current_ts, ACTION_ERROR);
  }


  /**
   * @brief Returns the configured number of checkpoints.
   *
   * Uniform checkpointing requires the number of checkpoints (`snaps`)
   * to be explicitly defined during construction.
   *
   * @return The configured number of checkpoints.
   */
  int getNumberOfCheckpoints() override {
    if (snaps == 0) {
          GPUZIPLogger::Error("There must be set a value for snapshots.\n");
    }
    return snaps;
  }
  
public:

  /**
   * @brief Constructor for the UniformCheckpointing class.
   *
   * @param steps The number of computational steps for which checkpointing is
   * required.
   * @param snaps Total number of checkpoints used by the algorithm.
   *
   * Initializes the base `Checkpointing` class and computes the uniformly
   * distributed checkpoint locations.
   */
  UniformCheckpointing(int steps, int snaps) 
      : Checkpointing(steps, snaps) { 
        setCheckpoints(); 
  }
};