#pragma once

#include <stdio.h>
#include <stdlib.h>

#include <map>
#include <string>
#include <vector>

#include <fstream>
#include <iostream>
#include <sstream>

#include "../Checkpointing.hpp"

#include "../../common/GPUZIPLogger.cpp"

/**
 * @class UniformCheckpointing
 * @brief Implements a checkpointing mechanism using uniform checkpoint spacing.
 * @author Bruno Ortega <brunoteixeira@estudante.ufscar.br>
 * @date May 26th, 2026
 *
 * This class extends the base `Checkpointing` class to provide specific
 * checkpointing actions (save, restore, forward, backward, terminate) using
 * a fixed-spacing checkpoint distribution strategy.
 *
 * The algorithm stores checkpoints at fixed timestep intervals and
 * recomputes forward states between restored checkpoints during the
 * adjoint phase.
 */
class UniformCheckpointing : public Checkpointing {

private:
  bool adjoint = false;    //< Indicates whether execution is currently in
                           // the forward or in the adjoint phase.
  bool save = false;       //< Indicates whether the forward computation for the current
                           // timestep has already been issued before saving the checkpoint.
  bool restore = false;    //< Control variable to allow backward and restore
                           // actions for the same timestep.
  int current_ts = 1;      //< Current timestep.
  int last_checkpoint = 0; //< Timestep corresponding to the last restored checkpoint.
  int adj_fwd_ts = 0;      //< Current timestep during forward recomputation
                           // in the adjoint phase.

protected:

  /**
   * @brief Resets the internal state of the checkpointing process.
   *
   * Sets `adjoint`, `save`, `restore`, `current_ts`, `last_checkpoint` and
   * `adj_fwd_ts` to their initial values.
   * This is typically called to reinitialize the checkpointing algorithm.
   */
  void reset() override {
    adjoint = false;
    save = false;
    restore = false;
    current_ts = 1;
    last_checkpoint = 0;
    adj_fwd_ts = 0;
  }

  /**
   * @brief Determines the next action to perform in the checkpointing process.
   *
   * @return An `Action` object describing the next step, including its type and
   * relevant parameters.
   */
  Action getAction() override {
    int spacing = std::max(1, steps / snaps);

    // Forward from first to last timestep
    if(!adjoint) {
      // At last timestep, forward finishes and adjoint begins
      if(current_ts == steps) {
        adjoint = true;
        return Action(current_ts, ACTION_FORWARD);
      }
      // Apply forward and save for the current timestep
      if(current_ts == 1 || current_ts % spacing == 0) {
        if(!save) {
          save = true;
          return Action(current_ts, ACTION_FORWARD);
        }
        save = false;
        last_checkpoint = current_ts;
        current_ts++;
        return Action(current_ts-1, ACTION_SAVE);
      }
      // Apply forward for the current timestep
      current_ts++;
      return Action(current_ts-1, ACTION_FORWARD);
    }
    // Adjoint from last to first timestep
    // Adjoint finishes
    if(current_ts == 0) {
      return Action(current_ts, ACTION_TERMINATE);
    }
    // Apply backward and recover last saved snapshot
    if(current_ts % spacing == 0 || current_ts == steps) {
      if(!restore) {
        restore = true;
        return Action(current_ts, ACTION_BACKWARD);
      }
      // Get last saved checkpoint timestep
      if(current_ts % spacing == 0)
        last_checkpoint = std::max(1, current_ts - spacing);
      adj_fwd_ts = last_checkpoint;
      restore = false;
      current_ts--;
      return Action(last_checkpoint, ACTION_RESTORE);
    }
    // Forward from restored checkpoint to current timestep
    if(adj_fwd_ts <= current_ts) {
      adj_fwd_ts++;
      return Action(adj_fwd_ts-1, ACTION_FORWARD);
    } else { // Backward at current timestep
      current_ts--;
      adj_fwd_ts = last_checkpoint;
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
   * Initializes the base `Checkpointing` class and sets up the internal state.
   */
  UniformCheckpointing(int steps, int snaps) 
      : Checkpointing(steps, snaps) {}
};